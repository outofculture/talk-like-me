from keras import Model, Input
from keras.activations import relu, softmax
from keras.layers import Dense
from keras.losses import categorical_crossentropy, binary_crossentropy
import numpy as np

import audio

discriminator_layers = Dense(1, activation=softmax)(audio.flattened_audio_input)
discriminator_model = Model(audio.flattened_audio_input, discriminator_layers)
discriminator_model.compile(optimizer='adadelta', loss=categorical_crossentropy)


noise_input = Input((100 + 10,))  # 100 random numbers
generator_layers = Dense(audio.flattened_input_shape, activation=relu)(noise_input)
generator_model = Model(noise_input, generator_layers)
generator_model.compile(optimizer='adadelta', loss=binary_crossentropy)

if __name__ == '__main__':
    data, labels = audio.load_digits(audio.sample_rate)
    shaped_data, deshaper = audio.audio_data_to_flattened_normalized_ffts(data)
    print('Discriminator:')
    discriminator_model.summary()
    print('\n\nGenerator:')
    generator_model.summary()

    discriminator_model.fit(
        shaped_data, np.ones((shaped_data.shape[0],), dtype=np.float32),
        shuffle=True,
        batch_size=3,
        epochs=3,
    )
