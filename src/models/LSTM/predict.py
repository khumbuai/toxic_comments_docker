from src.models.LSTM.preprocess import preprocess
from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
import tensorflow as tf
import numpy as np


class ToxicLSTMModel:

    def __init__(self):

        # this is key : save the graph after loading the model

        self.dir = 'src/models/LSTM/'
        self.model = load_model(self.dir + 'model.hdf5')
        self.graph = tf.get_default_graph()
        self.preprocess = preprocess
        self.tokenizer = self.load_tokenizer()
        self.maxlen = 150
        self.categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    def predict(self,text):
        X = self.transform_to_input(text)
        with self.graph.as_default():
            pred = self.model.predict(X)[0]


        pred = list(np.round(pred,2))
        result = {self.categories[i]:str(pred[i]) for i in range(len(pred))}


        return result

    def load_tokenizer(self):
        with open(self.dir + 'tokenizer.p','rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer

    def transform_to_input(self,text):
        text = self.preprocess(text)
        X = self.tokenizer.texts_to_sequences([text])
        X = pad_sequences(X, maxlen=self.maxlen)

        return X

#
if __name__ == '__main__':
    TM = ToxicLSTMModel()

    text = 'I like this donout very much'
    pred = TM.predict(text)
    print(pred)