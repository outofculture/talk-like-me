from functools import reduce
from math import ceil

import numpy as np
from keras import Input, Model
from keras.backend import categorical_crossentropy, sigmoid, relu
from keras.layers import Dense
from keras.optimizers import SGD
from scipy.fftpack import fft

from audio import load_digits

sample_rate = 8000

lowest_freq = 20  # hz
window_size = 1. / lowest_freq
samples_per_window = ceil(sample_rate * window_size)
window = np.hamming(samples_per_window // 2)
samples_per_step = ceil(sample_rate * 0.01)
steps = ceil(sample_rate / samples_per_step)

input_shape = (steps, samples_per_window // 2)  # (108, sample_rate)
input_size = reduce(lambda t, a: a * t, input_shape, 1)
image_input = Input(shape=input_shape)

####### Encode ########

# Micro features step
# Convolution of (1103, 96) into
# Conv2D(filters=10, kernel_size=(3, 3))()  # outputs (1101, 94, 10)
# MaxPooling2D(pool_size=(3, 3))()  # outputs (367, 31, 10)

# Global frequency step
# Conv2D(filters=100, kernel_size=(367, 1, 10))()  # outputs (1, 31, 100)
# MaxPooling2D(pool_size=(1, 10))()  # outputs (1, 3, 100)

# Compress with a fully-connected layer
encoded = Dense(1, activation=relu)(image_input)

####### Decode ########
decoded = Dense(input_size, activation=sigmoid)(encoded)  # outputs (1103, 96)

# UpSampling2D(size=(1, 10))()  # outputs (1, 30, 100)
# Conv2D(filters=367, kernel_size=(1, 3), padding='same')()  # outputs (1, 30, 367)

# Conv3D(filters=10, kernel_size=(1, 3, 3), padding='same')()
# UpSampling2D(size=(1, 3))()

# Conv2D(filters=1, kernel_size=(what))()

autoencoder = Model(image_input, decoded)
# encoder = Model(image_input, encoded)

autoencoder.compile(
    optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True),
    loss=categorical_crossentropy)


def this_never_worked():
    directory_listing = []  # TODO
    chunked_files = []

    for filename in directory_listing:
        with open(filename) as the_file:
            chunks = np.empty((1,) + input_shape)
            samples = the_file.read()
            file_len = len(samples)
            assert file_len == sample_rate
            for i in range(chunks.shape[1]):
                start = i * samples_per_step
                this_chunk = samples[start:start + samples_per_window]
                chunks[0, i] = np.abs(fft(this_chunk))[:chunks.shape[2]]
            chunked_files.append(chunks)
    chunked_files = np.concatenate(chunked_files, axis=0)


def shape_data_for_processing():
    data, labels = load_digits(sample_rate)
    chunked_data = []
    for spoken_word in data:
        chunks = np.empty(input_shape)
        for i in range(steps):
            step_start = i * samples_per_step
            range_start = ceil(step_start - ((samples_per_window - samples_per_step) / 2))
            range_end = range_start + samples_per_window
            if range_start < 0 or range_end > sample_rate:
                continue
            chunks[i] = window * np.abs(fft(
                spoken_word[range_start:range_end]))[:samples_per_window // 2]
        chunked_data.append(chunks)
    # return np.concatenate(chunked_data)
    return chunked_data


all_data = shape_data_for_processing()
autoencoder.fit(all_data, all_data)

prediction = autoencoder.predict(all_data)
error = ((prediction - all_data) ** 2).mean() ** 0.5
print(error)
# TODO play back the prediction (avg the ffts?)
