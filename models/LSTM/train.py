import pandas as pd
import numpy as np
#todo change to relative path
from models.LSTM.preprocess import preprocess
from tqdm import tqdm
tqdm.pandas()
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# config

EMBEDDING_FILE = "assets/crawl-300d-2M.vec"
TRAIN_FILENAME = "assets/train.csv"
categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
embed_size = 300
max_features = 150000
max_text_len = 150

train = pd.read_csv(TRAIN_FILENAME).sample(frac=0.1)
y = train[categories].values

train["comment_text"].fillna("no comment", inplace = True)
train["comment_text"] = train["comment_text"].progress_apply(lambda x: preprocess(x))

tk = Tokenizer(num_words = max_features, lower = True)
tk.fit_on_texts(train["comment_text"])

X = tk.texts_to_sequences(train["comment_text"])
X = pad_sequences(X,maxlen=max_text_len)

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.1,random_state=23)

def get_coefs(word,*arr): return word, np.asarray(arr, dtype = "float32")
embeddings_index = dict(get_coefs(*o.strip().split()) for o in tqdm(open(EMBEDDING_FILE)))

word_index = tk.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in tqdm(word_index.items()):
    if i >= nb_words: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


from models.LSTM.model import Lstm

model = Lstm(maxlen=max_text_len,max_features=max_features,embed_size=embed_size)
model.compile(loss = "binary_crossentropy", optimizer = 'adam',  metrics = ["accuracy"])

