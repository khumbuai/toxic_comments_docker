from keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints


def Lstm(maxlen, max_features, embed_size, emb_matrix):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size,
                  trainable=False, weights=[emb_matrix])(inp)
    x = LSTM(64, return_sequences=True)(x)
    x1 = GlobalAveragePooling1D()(x)
    x2 = GlobalMaxPooling1D()(x)
    x = Concatenate()([x1,x2])
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=out)

    return model
