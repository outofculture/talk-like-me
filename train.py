from math import ceil

import numpy as np
from keras import Input, Model
from keras.backend import categorical_crossentropy, sigmoid, relu
from keras.layers import Dense
from keras.optimizers import SGD
from scipy.fftpack import fft

sample_rate = 44100
lowest_freq = 20  # hz
window_size = 1. / lowest_freq
samples_per_window = ceil(sample_rate * window_size)
samples_per_step = ceil(sample_rate * 0.01)

input_shape = (96, 1103)
image_input = Input(shape=input_shape)

####### Encode ########

# Micro features step
# Convolution of (1103, 96) into
# model.add(Conv2D(filters=10, kernel_size=(3, 3)))  # outputs (1101, 94, 10)
# model.add(MaxPooling2D(pool_size=(3, 3)))  # outputs (367, 31, 10)

# Global frequency step
# model.add(Conv2D(filters=100, kernel_size=(367, 1, 10)))  # outputs (1, 31, 100)
# model.add(MaxPooling2D(pool_size=(1, 10)))  # outputs (1, 3, 100)

# Compress with a fully-connected layer
encoded = Dense((20,), activation=relu)(image_input)

# Decode
decoded = Dense(input_shape, activation=sigmoid)(encoded)  # outputs (1103, 96)

# model.add(UpSampling2D(size=(1, 10)))  # outputs (1, 30, 100)
# model.add(Conv2D(filters=367, kernel_size=(1, 3), padding='same'))  # outputs (1, 30, 367)

# model.add(Conv3D(filters=10, kernel_size=(1, 3, 3), padding='same'))
# model.add(UpSampling2D(size=(1, 3)))

# model.add(Conv2D(filters=1, kernel_size=()))

autoencoder = Model(image_input, decoded)
encoder = Model(image_input, encoded)

autoencoder.compile(
    optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True),
    loss=categorical_crossentropy)

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

autoencoder.fit(chunked_files, chunked_files)

prediction = autoencoder.predict(chunked_files)
error = ((prediction - chunked_files) ** 2).mean() ** 0.5
print(error)
