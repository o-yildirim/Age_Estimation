import tensorflow as tf
from keras.layers import *
from keras import regularizers, Model

def get_model(input_shape):
    inputs = Input(shape=input_shape)

    conv_0 = Conv2D(64, (7, 7), activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal(),
                    bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.00001))(inputs)

    bn0 = BatchNormalization()(conv_0)
    max_pool0 = MaxPool2D(pool_size=(2, 2), padding='valid')(bn0)


    conv_1 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal(),
                    bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.00001))(max_pool0)

    bn1 = BatchNormalization()(conv_1)
    max_pool1 = MaxPool2D(pool_size=(2, 2), padding='valid')(bn1)

    conv_2 = Conv2D(128, (2, 2), activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal(),
                    bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.00001))(max_pool1)
    bn2 = BatchNormalization()(conv_2)

    conv3 = Conv2D(128, (1, 1), activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal(),
                   bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.00001))(bn2)

    flatten_1 = Flatten()(conv3)
    dropout_1 = Dropout(0.6)(flatten_1)

    regression_output = Dense(units=1, activation='relu')(dropout_1)

    model = Model(inputs=inputs, outputs=regression_output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=tf.keras.metrics.MeanAbsoluteError())
    model.summary()
    return model
