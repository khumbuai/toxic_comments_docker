from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import CuDNNLSTM, Bidirectional, Dropout, TimeDistributed, concatenate, Lambda
import keras.backend as K

from src.models.LSTM_MultiAttention.custom_layers import Attention


def LSTM_MultiAttentions(maxlen, max_features, embed_size, embedding_matrix):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix],
                  trainable=False)(inp)
    x = Bidirectional(CuDNNLSTM(300, return_sequences=True))(x)

    attention_layers = [Attention(maxlen) for _ in range(6)]
    attentions = [attention_layer(x) for attention_layer in attention_layers]
    attentions = [Lambda(lambda x: K.expand_dims(x, axis=-1))(attention) for attention in attentions]

    x = concatenate(attentions, axis=-1)
    x = Lambda(lambda s: K.permute_dimensions(s, (0, 2, 1)))(x)
    x = TimeDistributed(Dense(64, activation="relu"))(x)
    x = TimeDistributed(Dropout(0.125))(x)
    x = TimeDistributed(Dense(1, activation="sigmoid"))(x)
    x = Lambda(lambda x: K.squeeze(x, axis=-1))(x)
    model = Model(inputs=inp, outputs=x)

    return model, [attention_layer.attention_weights for attention_layer in attention_layers]


if __name__ == '__main__':
    import numpy as np

    embedding_matrix = np.zeros((100000, 300))
    model, attention_weights = LSTM_MultiAttentions(maxlen=150, max_features=100000, embedding_matrix=embedding_matrix, embed_size=300)
    model.summary()
    print(attention_weights)
