from __future__ import division

import gzip
import os
import wave
from functools import reduce
from math import ceil

import numpy as np
import pyaudio
import scipy.signal
from keras import Input
from scipy.signal import stft, istft

from utils import sliding_window


def load_digits(sample_rate=8000., validation=False, random=False):
    cache_file = 'digits_%dkHz.npy' % (sample_rate // 1000)
    if not os.path.exists(cache_file):
        print("Resampling to %d kHz..." % (sample_rate // 1000))
        wav_file = wave.open(gzip.open('digits.wav.gz', 'rb'), 'rb')
        assert wav_file.getsampwidth() == 2

        # read wav data
        raw_sample_rate = wav_file.getframerate()
        n_samples = wav_file.getnframes()
        raw_data = wav_file.readframes(n_samples)
        data = np.fromstring(raw_data, dtype=np.int16)

        # resample
        downsampled_size = int(n_samples * sample_rate / raw_sample_rate)
        data = scipy.signal.resample(data, downsampled_size)

        # crop and reshape to (n_digits, digit_size)
        digit_duration = 1.0
        digit_size = int(digit_duration * sample_rate)
        n_digits = downsampled_size // digit_size
        cropped_size = n_digits * digit_size
        data = data[:cropped_size].reshape(n_digits, digit_size)

        # fade in/out to get rid of poppyclicks
        ramp = np.linspace(0, 1, int(0.05 * sample_rate))
        data[:, :len(ramp)] *= ramp[None, :]
        data[:, -len(ramp):] *= ramp[None, ::-1]

        # convert back to int16 to reduce size
        data = data.astype('uint16')

        # cache it
        np.save(open(cache_file, 'wb'), data)
        print("   saved cache to " + cache_file)
    else:
        # reload from cache
        data = np.load(open(cache_file, 'rb'))

    bad = [23, 100]  # oops
    for i in bad:
        data[i] = data[i + 10]

    labels = np.arange(2, 2 + data.shape[0]) % 10

    if random:
        # randomize the order
        order = np.arange(len(data))
        np.random.shuffle(order)
        data = data[order]
        labels = labels[order]

    if validation:
        return data[:-20], labels[:-20], data[-20:], labels[-20:]
    else:
        return data, labels


pyaudio_handle = None
pyaudio_stream = None


def play(data, sample_rate):
    global pyaudio_stream, pyaudio_handle
    data = data.astype('int16')
    if pyaudio_handle is None:
        pyaudio_handle = pyaudio.PyAudio()
        pyaudio_stream = pyaudio_handle.open(
            format=pyaudio_handle.get_format_from_width(data.itemsize),
            channels=1,
            rate=sample_rate,
            output=True,
            frames_per_buffer=1000,
        )
    padding = np.zeros(int(sample_rate * 0.5), dtype=data.dtype)
    data = np.concatenate([data, padding])
    pyaudio_stream.write(data)
    # time.sleep(len(data) / sample_rate)


def play_all(data, labels, sample_rate):
    for i in range(len(data)):
        print(labels[i])
        play(data[i], sample_rate)


sample_rate = 16000
lowest_freq = 20  # hz
window_size = 1. / lowest_freq
samples_per_window = int(ceil(sample_rate * window_size))
window = np.hamming(samples_per_window // 2)
samples_per_step = int(ceil(sample_rate * 0.01))
steps = ceil(sample_rate / samples_per_step)
input_shape = (
    (samples_per_window // 2) + 1,
    ((sample_rate // samples_per_step) + 1) * 2,
)
fft_audio_input = Input(shape=input_shape)
flattened_input_shape = reduce(lambda t, a: a * t, input_shape, 1)
flattened_audio_input = Input(shape=(flattened_input_shape,))


def audio_data_to_expectation_of_next_sample(data, labels):
    orig_type = data.dtype
    data_max = data.max()
    data = data / data_max
    expectations = data
    window_size = 512
    padding = np.zeros((data.shape[0], window_size), dtype=data.dtype)
    data = np.concatenate((padding, data), axis=1)

    def denormalize(normalized_data):
        return (normalized_data * data_max).astype(orig_type)

    chunks = sliding_window(data, size=window_size)[:, :-1]  # drop the last one, which has no prediction
    # todo include the label data somehow
    # todo maybe also include the index of the next sample?
    return chunks, expectations, denormalize


def audio_data_to_windows_of_normalized_ffts(data):
    ffts = stft(
        data,
        sample_rate,
        nperseg=samples_per_window,
        noverlap=samples_per_step * 4,
    )[2]
    coefs_only = np.dstack((ffts.real, ffts.imag))
    mean = coefs_only.mean()
    centered_on_zero = coefs_only - mean
    the_max = np.abs(centered_on_zero).max()
    normalized_on_zero = centered_on_zero / the_max
    normalized_between_zero_and_one = (normalized_on_zero + 1) / 2

    def deshaper(processed_data):
        denormalized = (((processed_data * 2) - 1) * the_max) + mean
        result = np.empty(ffts.shape, dtype=complex)
        result.real, result.imag = np.dsplit(denormalized, 2)
        return istft(
            result,
            sample_rate,
            nperseg=samples_per_window,
            noverlap=samples_per_step * 4,
        )[1]

    return normalized_between_zero_and_one, deshaper


def audio_to_spectrogram(data):
    return stft(
        data,
        sample_rate,
        nperseg=samples_per_window,
        noverlap=samples_per_step * 4,
    )[2]


def spectrogram_to_audio(data):
    return istft(
        data,
        sample_rate,
        nperseg=samples_per_window,
        noverlap=samples_per_step * 4,
    )[1]


def audio_data_to_flattened_normalized_ffts(data):
    normalized_between_zero_and_one, deshaper = audio_data_to_windows_of_normalized_ffts(data)
    flattened = normalized_between_zero_and_one.reshape((data.shape[0], flattened_input_shape))

    def deflatten_and_deshape(processed_data):
        expanded = processed_data.reshape((data.shape[0],) + input_shape)
        return deshaper(expanded)

    return flattened, deflatten_and_deshape


def reduce_audio_data_fidelity(data, quantization_steps=256):
    data_max = (1 << 15) - 1
    data = data / data_max
    # mu-law companding transformation (ITU-T, 1988)
    mu = quantization_steps - 1
    data = np.sign(data) * np.log(1 + mu * np.absolute(data)) / np.log(1 + mu)

    def deshaper(processed_data):
        return processed_data * 65535

    # quantize
    return (np.clip(data * 0.5 + 0.5, 0, 1) * mu).astype(np.int32), deshaper


if __name__ == '__main__':
    data, labels = load_digits(sample_rate, random=True)

    chunks, expectations, deshaper = audio_data_to_expectation_of_next_sample(data, labels)

    # play(data[0], sample_rate)
    # play_all(data, labels, sample_rate)
