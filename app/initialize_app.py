import numpy as np
import pickle
from collections import defaultdict
import nltk

from backend_flask.model import BidLstm, evaluate_attention


def predict(model, attention_weights, word_index):

    def predict_sequence(text, length=150):
        text = text.lower()
        text = nltk.word_tokenize(text)
        filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        text = text[:length]
        input_seq = np.array([0 for _ in range(length - len(text))] + [word_index[word] for word in text])
        attentions = evaluate_attention(model, attention_weights, np.expand_dims(input_seq, axis=0))
        predictions = model.predict(np.expand_dims(input_seq, axis=0))
        attentions = attentions[0, -len(text):, 0]
        return predictions, attentions

    return predict_sequence


def initialize():
    with open('assets/word_index.p', 'rb') as handle:
        word_index = pickle.load(handle)

    word_index = defaultdict(lambda: 1, word_index)
    model, attention_weights = BidLstm(maxlen=150, max_features=100000, embed_size=300)
    model.load_weights('assets/model.hdf5')
    model.summary()
    predict_on_sequence = predict(model, attention_weights, word_index)
    return predict_on_sequence


def order_by_attention(sentence, attention):
    sentence = sentence.split(' ')
    return [word for _, word in sorted(zip(attention, sentence))]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    predict_on_sequence = initialize()
    sentence = r'''
    When I first met you, I was fascinated of you.
    Then I noticed that you are really a dishonest person, disgusting'''
    # [[5.3925413e-01 1.9366432e-04 1.6936582e-02 2.6108450e-04 1.9309042e-01 2.6494547e-04]]
    sentence1 = r'''
    When I first met you, I was not fascinated of you.
    Then I noticed that you are not really a dishonest person, you were just disgusted because of the food'''
    #[[9.6523799e-02 1.3626341e-05 1.0345851e-03 2.8348686e-05 9.4220955e-03 7.2261028e-05]]

    toxicity, attention = predict_on_sequence(sentence)
    print(toxicity)
    print(attention)
    print(order_by_attention(sentence, attention))


    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult",
                    "identity_hate"]

    plt.plot(range(len(attention)), attention)
    plt.show()

    fig = plt.figure()

    ax = fig.add_subplot(111)

    x = 0
    y = 0
    for i, word in enumerate(nltk.word_tokenize(sentence)[:150]):
        x += 10 * len(word)
        if x > 400:
            x = 0
            y -= 2
        ax.text(x, y,  word, bbox={'facecolor':'red', 'alpha': attention[i] * 10, 'pad': 10},
                color='green', fontsize=10)

    ax.axis([0, 400, y, 0])
    plt.show()
