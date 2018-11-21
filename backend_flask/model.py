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
    attention_layer = Attention(maxlen)
    x = attention_layer(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)

    return model, attention_layer.attention_weights


def evaluate_attention(model, attention_weights, input_sequence):
    sess = K.get_session()
    return sess.run(attention_weights, feed_dict={model.input: input_sequence})


if __name__ == '__main__':
    model, attention_weights = BidLstm(maxlen=150, max_features=100000, embed_size=300)
    model.load_weights('assets/model.hdf5')
    model.summary()

    input_sequence = np.array([[i for i in range(150)]])

    print(evaluate_attention(model, attention_weights, input_sequence))