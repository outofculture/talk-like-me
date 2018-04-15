from comet_ml import Experiment

from keras import Model, Input
from keras.backend import categorical_crossentropy, sigmoid, relu, binary_crossentropy
from keras.layers import Dense
from keras.optimizers import SGD

from audio import load_digits, sample_rate, input_shape, image_input, \
    shape_data_for_processing, flattened_input_shape

try:
    from local_settings import COMET_API_KEY
    experiment = Experiment(api_key=COMET_API_KEY, project_name="talk-like-me")
except ImportError:
    experiment = None

####### Encode ########

# Micro features step
# Convolution of (1103, 96) into
# Conv2D(filters=10, kernel_size=(3, 3))()  # outputs (1101, 94, 10)
# MaxPooling2D(pool_size=(3, 3))()  # outputs (367, 31, 10)

# Global frequency step
# Conv2D(filters=100, kernel_size=(367, 1, 10))()  # outputs (1, 31, 100)
# MaxPooling2D(pool_size=(1, 10))()  # outputs (1, 3, 100)

# Compress with a fully-connected layer
# encoded = Dense(input_shape[1], activation=relu)(image_input)
encoding_dim = 189
encoded = Dense(encoding_dim, activation=relu)(image_input)

####### Decode ########
decoded = Dense(flattened_input_shape, activation=sigmoid)(encoded)

# UpSampling2D(size=(1, 10))()  # outputs (1, 30, 100)
# Conv2D(filters=367, kernel_size=(1, 3), padding='same')()  # outputs (1, 30, 367)

# Conv3D(filters=10, kernel_size=(1, 3, 3), padding='same')()
# UpSampling2D(size=(1, 3))()

# Conv2D(filters=1, kernel_size=(what))()

autoencoder = Model(image_input, decoded)
encoder = Model(image_input, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(
    # optimizer='adadelta',
    # loss='mean_squared_error',
    # loss=binary_crossentropy,
    optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True),
    loss=categorical_crossentropy,
)

if __name__ == '__main__':
    data, labels = load_digits(sample_rate)
    shaped_data, deshaper = shape_data_for_processing(data)
    # play_all(deshaper(shaped_data), labels, sample_rate)
    # zeros = np.zeros(shaped_data.shape)
    autoencoder.fit(
        shaped_data, shaped_data,
        shuffle=True,
        batch_size=3,
        epochs=5,
        # validation_data=(shaped_data, shaped_data),
    )
    # autoencoder.fit(zeros, zeros)

    # prediction = autoencoder.predict(zeros)
    prediction = autoencoder.predict(shaped_data)
    error = ((prediction - shaped_data) ** 2).mean() ** 0.5
    print('mean squared error: {}'.format(error))
    # play_all(deshaper(prediction), labels, sample_rate)
