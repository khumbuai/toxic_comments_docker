import numpy as np

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, Dropout, CuDNNLSTM
import keras.backend as K

from src.models.LSTMAttention.custom_layers import Attention


def LSTMAttention(maxlen, max_features, embedding_matrix, embed_size):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size,
                  trainable=False)(inp)
    x = Bidirectional(LSTM(300, return_sequences=True, dropout=0.25, weights=[embedding_matrix],
                           recurrent_dropout=0.25))(x)
    attention_layer = Attention(maxlen)
    x = attention_layer(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)

    return model, attention_layer.attention_weights


if __name__ == '__main__':
    embedding_matrix = np.zeros((100000, 300))
    model, attention_weights = LSTMAttention(maxlen=150, max_features=100000,
                                             embedding_matrix=embedding_matrix, embed_size=300)
    #model.load_weights('assets/model.hdf5')
    model.summary()
