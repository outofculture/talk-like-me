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
samples_per_step = ceil(sample_rate * 0.01)

input_shape = (96, 1103)
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
encoded = Dense((20,), activation=relu)(image_input)

####### Decode ########
decoded = Dense(input_shape, activation=sigmoid)(encoded)  # outputs (1103, 96)

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


data, labels = load_digits(sample_rate)

autoencoder.fit(data, data)

prediction = autoencoder.predict(data)
error = ((prediction - data) ** 2).mean() ** 0.5
print(error)
# TODO play back the prediction
