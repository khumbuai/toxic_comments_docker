'''
Input expected from frontend js:

var input_text = '"input_text": ["Hate", "you", "idiot!"],';
var probabilities = '"probabilities": [' +  toxicity.join(", ") + '],';
var toxic_labels = '"toxic_labels": ["Toxic", "Severe", "Obscene", "Threat", "Insult", "Identity"],';
var attentions ='"attn_dists": [[1, 0.2, 0], [0, 0.8, 0.3],' +
                '[0.2, 0.4, 1], [0, 0.2, 0.3],' +
                '[1, 0.2, 0], [1, 0.8, 0.9]]';
var json_file ='{' + input_text + probabilities + toxic_labels + attentions + '}';
'''
import numpy as np
import pickle

import keras.backend as K
from keras.preprocessing.sequence import pad_sequences

from src.models.LSTMAttention.preprocess import preprocess
from src.models.LSTMAttention.config import maxlen, max_features, embed_size
from src.models.LSTM_MultiAttention.model import LSTM_MultiAttentions


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class ToxicMultiAttentionModel():
    def __init__(self, dir='./'):
        self.dir = dir
        self.model, self.attention_weights = LSTM_MultiAttentions(maxlen, max_features, embed_size,
                                                                  embedding_matrix=np.zeros((max_features, 300)))
        self.model.load_weights(self.dir + 'model.hdf5')
        self.model.summary()
        self.preprocess = preprocess
        self.tokenizer = self.load_tokenizer()
        self.maxlen = maxlen
        self.categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    def load_tokenizer(self):
        with open(self.dir + 'tokenizer.p', 'rb') as f:
                tokenizer = pickle.load(f)
        return tokenizer

    def evaluate_attention(self, input_sequence):
        sess = K.get_session()
        return sess.run(self.attention_weights, feed_dict={self.model.input: input_sequence})

    def transform_to_input(self, text):
        X = self.tokenizer.texts_to_sequences([text])
        X = pad_sequences(X, maxlen=self.maxlen)
        return X

    def predict(self, raw_text):
        text = self.preprocess(raw_text)
        input_seq = self.transform_to_input(text)

        words = text.split(' ')
        preds_as_dict = self._predict_toxicity(input_seq)
        attentions_as_dict = self._predict_attentions(input_seq, words)
        words_as_dict = {'input_text': words}
        # Make sure that the return value does not conatin any numpy datatype (such as numpy.float32),
        # since flask cannot jsonify it
        return {**words_as_dict, ** preds_as_dict, **attentions_as_dict}

    def _predict_toxicity(self, input_seq):
        pred = self.model.predict(input_seq)[0]
        pred = np.round(pred, 2).tolist()
        preds_as_dict = {'probabilities': pred}
        return preds_as_dict

    def _trim_attentions_to_correct_length(self, attentions, words):
        trimmed_attentions = []
        for attention in attentions:
            trimmed_attentions.append(attention[0, -len(words):, 0].tolist())
        return trimmed_attentions

    def _cool_down(self, attentions, T=0.001):
        cooled_attentions = []
        for attention in attentions:
            cooled_attentions.append(softmax(np.array(attention) / T))
        return cooled_attentions

    def _convert_2_list(self, attentions):
        for i, attention in enumerate(attentions):
            attentions[i] = list(attention)
        return attentions

    def _predict_attentions(self, input_seq, words):
        attentions = self.evaluate_attention(input_seq)
        # grab the last attions, note that we pre-padded the input sequence
        attentions = self._trim_attentions_to_correct_length(attentions, words)
        attentions = self._cool_down(attentions)
        attentions = self._convert_2_list(attentions)

        return {"attn_dists": attentions}


if __name__ == '__main__':

    TM = ToxicMultiAttentionModel()
    text = r'''	
    What the fuck are you doing?	
     When I first met you, I was fascinated of you.		
     Then I noticed that you are really a dishonest person, disgusting'''
    text_ = r'''		
     When I first met you, I was not fascinated of you.		
     Then I noticed that you are not really a dishonest person, you were just disgusted because of the food'''

    pred = TM.predict(text)
    print(pred)
