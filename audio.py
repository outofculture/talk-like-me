from __future__ import division
import os, wave, gzip
import numpy as np
import pyqtgraph as pg
import scipy.signal
import pyaudio
import time


def load_digits(sample_rate=8000):
    cache_file = 'digits_%dkHz.npy' % (sample_rate//1000)
    if not os.path.exists(cache_file):
        print("Resampling to %d kHz..." % (sample_rate//1000))
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
        ramp = np.linspace(0, 1, 0.05 * sample_rate)
        data[:,:len(ramp)] *= ramp[None,:]
        data[:,-len(ramp):] *= ramp[None,::-1]
        
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
        data[i] = data[i+10]

    labels = np.arange(2, 2 + data.shape[0]) % 10
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
    padding = np.zeros(int(sample_rate*0.5), dtype=data.dtype)
    data = np.concatenate([data, padding])
    pyaudio_stream.write(data)
    #time.sleep(len(data) / sample_rate)


def play_all(data, labels, sample_rate):
    for i in range(len(data)):
        print(labels[i])
        play(data[i], sample_rate)


if __name__ == '__main__':
    sample_rate = 8000
    data, labels = load_digits(sample_rate)

    # randomize the order
    order = np.arange(len(data))
    np.random.shuffle(order)
    data = data[order]
    labels = labels[order]

    #play(data[0], sample_rate)
    #play_all(data, labels, sample_rate)
    
    
    
    