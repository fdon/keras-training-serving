mport tensorflow as tf
IM_SIZE = 16

image_input = tf.keras.Input(shape=(IM_SIZE, IM_SIZE, 3))

# Some convolutional layers
conv_1 = tf.keras.layers.Conv2D(32,
                                kernel_size=(3, 3),
                                padding='same',
                                activation='relu')(image_input)
conv_1 = tf.keras.layers.MaxPooling2D(padding='same')(conv_1)
conv_2 = tf.keras.layers.Conv2D(32,
                                kernel_size=(3, 3),
                                padding='same',
                                activation='relu')(conv_1)
conv_2 = tf.keras.layers.MaxPooling2D(padding='same')(conv_2)

# Flatten the output of the convolutional layers
conv_flat = tf.keras.layers.Flatten()(conv_2)

# Some dense layers with two separate outputs
fc_1 = tf.keras.layers.Dense(128,
                             activation='relu')(conv_flat)
fc_1 = tf.keras.layers.Dropout(0.2)(fc_1)
fc_2 = tf.keras.layers.Dense(128,
                             activation='relu')(fc_1)
fc_2 = tf.keras.layers.Dropout(0.2)(fc_2)

# Output layers: separate outputs for the weather and the ground labels
weather_output = tf.keras.layers.Dense(4,
                                       activation='softmax',
                                       name='weather')(fc_2)
ground_output = tf.keras.layers.Dense(13,
                                      activation='sigmoid',
                                      name='ground')(fc_2)
