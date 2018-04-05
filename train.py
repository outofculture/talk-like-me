from functools import reduce
from math import ceil

import numpy as np
from keras import Input, Model
from keras.backend import categorical_crossentropy, sigmoid, relu
from keras.layers import Dense
from keras.optimizers import SGD
from scipy.signal import stft, istft

from audio import load_digits, play_all

sample_rate = 8000

lowest_freq = 20  # hz
window_size = 1. / lowest_freq
samples_per_window = ceil(sample_rate * window_size)
window = np.hamming(samples_per_window // 2)
samples_per_step = ceil(sample_rate * 0.01)
steps = ceil(sample_rate / samples_per_step)

input_shape = (
    (samples_per_window // 2) + 1,
    ((sample_rate // samples_per_step) + 1) * 2,
)
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
encoded = Dense(32, activation=relu)(image_input)
# TODO try not compressing the data at all

####### Decode ########
decoded = Dense(input_shape[1], activation=sigmoid)(encoded)

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


def shape_data_for_processing(data):
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
    normalized = centered_on_zero / the_max

    def deshaper(processed_data):
        denormalized = (processed_data * the_max) + mean
        result = np.empty(ffts.shape, dtype=complex)
        result.real, result.imag = np.dsplit(denormalized, 2)
        return istft(
            result,
            sample_rate,
            nperseg=samples_per_window,
            noverlap=samples_per_step * 4,
        )[1]

    return normalized, deshaper


if __name__ == '__main__':
    data, labels = load_digits(sample_rate)
    shaped_data, deshaper = shape_data_for_processing(data)
    # play_all(deshaper(shaped_data), labels, sample_rate)
    # zeros = np.zeros(shaped_data.shape)
    autoencoder.fit(shaped_data, shaped_data)
    # autoencoder.fit(zeros, zeros)

    # prediction = autoencoder.predict(zeros)
    prediction = autoencoder.predict(shaped_data)
    error = ((prediction - shaped_data) ** 2).mean() ** 0.5
    print('mean squared error: {}'.format(error))
    play_all(deshaper(prediction), labels, sample_rate)
