import numpy as np
import pickle
import matplotlib.pyplot as plt

import keras.backend as K
from keras.preprocessing.sequence import pad_sequences

from src.models.LSTMAttention.model import LSTMAttention
from src.models.LSTMAttention.preprocess import preprocess
from src.models.LSTMAttention.config import maxlen, max_features, embed_size


class ToxicAttentionModel():
    def __init__(self, dir='./'):
        self.dir = dir
        self.model, self.attention_weights = LSTMAttention(maxlen, max_features, embed_size)
        self.model.load_weights(self.dir + 'model.hdf5')
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

    def predict(self, text):
        text = self.preprocess(text)
        words = text.split(' ')

        input_seq = self.transform_to_input(text)
        attentions = self.evaluate_attention(input_seq)
        # grab the last attions, note that we pre-padded the input sequence
        attentions = attentions[0, -len(words):, 0]

        pred = self.model.predict(input_seq)[0]
        pred = list(np.round(pred, 2))

        preds_as_dict = {self.categories[i]: str(pred[i]) for i in range(len(pred))}

        attentions_as_dict = {words[i]: str(attention) for i, attention in enumerate(attentions)}

        #TODO: which return values makes sense?
        return {** preds_as_dict, **attentions_as_dict}


    def visualize_attentions(self, text):
        pred = self.predict(text)

        words = self.preprocess(text).split(' ')
        attentions = [pred[word] for word in words]
        plt.plot(range(len(attentions)), attentions)
        plt.show()


if __name__ == '__main__':

    TM = ToxicAttentionModel()
    text = r'''	
    What the fuck are you doing?	
     When I first met you, I was fascinated of you.		
     Then I noticed that you are really a dishonest person, disgusting'''
    text_ = r'''		
     When I first met you, I was not fascinated of you.		
     Then I noticed that you are not really a dishonest person, you were just disgusted because of the food'''

    pred = TM.predict(text)
    print(pred)

    TM.visualize_attentions(text)
