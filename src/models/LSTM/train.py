import pandas as pd
import numpy as np
#todo change to relative path
from src.models.LSTM.preprocess import preprocess
from tqdm import tqdm
tqdm.pandas()
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from src.models.LSTM.model import Lstm
from keras.callbacks import ModelCheckpoint

# config

EMBEDDING_FILE = "src/assets/crawl-300d-2M.vec"
TRAIN_FILENAME = "src/assets/train.csv"
categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
embed_size = 300
max_features = 50000
maxlen = 150

train = pd.read_csv(TRAIN_FILENAME).sample(frac=0.1)
y = train[categories].values

train["comment_text"].fillna("no comment", inplace = True)
train["comment_text"] = train["comment_text"].progress_apply(lambda x: preprocess(x))

tk = Tokenizer(num_words = max_features, lower = True)
tk.fit_on_texts(train["comment_text"])
import pickle

with open('src/models/LSTM/tokenizer.p','wb') as f:
    pickle.dump(tk,f)
X = tk.texts_to_sequences(train["comment_text"])
X = pad_sequences(X,maxlen=maxlen)

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



model = Lstm(maxlen=maxlen,max_features=max_features, embedding_matrix=embedding_matrix, embed_size=embed_size)
model.summary()
model.compile(loss = "binary_crossentropy", optimizer = 'adam',  metrics = ["accuracy"])

ckpt = ModelCheckpoint('src/models/LSTM/model.hdf5',save_best_only=True, verbose=True)

model.fit(X_train, y_train, validation_data=(X_valid,y_valid), epochs=1, batch_size=256,callbacks=[ckpt],verbose=True)
