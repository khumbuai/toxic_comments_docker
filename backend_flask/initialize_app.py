import numpy as np
import pickle

from backend_flask.model import BidLstm, get_attention, calculate_attention


def predict(model, word_index):
    def predict_sequence(text):
        input_seq = text2seq(text, word_index, length=150)

        if input_seq.shape == 1:
            input_seq = np.expand_dims(input_seq, axis=0)

        predictions = model.predict(np.expand_dims(input_seq, axis=0))
        attention = get_attention(model)
        before, after = attention([input_seq])
        attention_weight = calculate_attention(before, after)
        return predictions, attention_weight

    return predict_sequence


def text2seq(text, word_index, length=150):
    text = text[:length]
    return np.array([word_index[word] for word in text] + [0 for _ in range(length- text)])


def initialize():
    with open('filename.pickle', 'rb') as handle:
        word_index = pickle.load(handle)

    model = BidLstm(maxlen=150, max_features=100000, embed_size=300)
    model.load_weights('weights.hdf5')
    model.summary()
    predict_on_sequence = predict(model, word_index)

    return predict_on_sequence
