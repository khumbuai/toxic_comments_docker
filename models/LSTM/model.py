from keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalAveragePooling1D
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints


def Lstm(maxlen, max_features, embed_size):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size,
                  trainable=False)(inp)
    x = LSTM(300, return_sequences=True)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)

    return model
