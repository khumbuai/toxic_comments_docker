import numpy as np

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, Dropout
import keras.backend as K

from backend_flask.custom_layers import Attention


def BidLstm(maxlen, max_features, embed_size):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size,
                  trainable=False)(inp)
    x = Bidirectional(LSTM(300, return_sequences=True, dropout=0.25,
                           recurrent_dropout=0.25))(x)
    x = Attention(maxlen)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)

    return model


def get_attention(model):
    before_attention = model.get_layer('bidirectional_1').output
    after_attention = model.get_layer('attention_1').output
    attention = K.function(model.inputs, [before_attention, after_attention])
    return attention


def calculate_attention(before, after):
    return [aft/bef for aft, bef if bef != 0 else 0 in zip(after, before)]


if __name__ == '__main__':
    model = BidLstm(maxlen=150, max_features=100000, embed_size=300)
    model.load_weights('weights.hdf5')
    model.summary()

    attention = get_attention(model)
    before, after = attention(np.array([[i for i in range(150)]]))
    print(calculate_attention(before, after))
