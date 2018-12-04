import pandas as pd
import numpy as np
import pickle

from tqdm import tqdm


tqdm.pandas()

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint

from src.models.LSTMAttention.preprocess import preprocess
from src.models.LSTM_MultiAttention.model import LSTM_MultiAttentions
from src.models.LSTMAttention.config import TRAIN_FILENAME, categories, maxlen, max_features, EMBEDDING_FILE,\
    embed_size, TOKENIZER_FILENAME, MODEL_FILENAME


train = pd.read_csv(TRAIN_FILENAME)
y = train[categories].values
train["comment_text"].fillna("no comment", inplace=True)
train["comment_text"] = train["comment_text"].progress_apply(lambda x: preprocess(x))

tk = Tokenizer(num_words=max_features, lower=False)
tk.fit_on_texts(train["comment_text"])

with open(TOKENIZER_FILENAME, 'wb') as f:
    pickle.dump(tk,f)
X = tk.texts_to_sequences(train["comment_text"])
X = pad_sequences(X,maxlen=maxlen)

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


model, _ = LSTM_MultiAttentions(maxlen=maxlen, max_features=max_features, embedding_matrix=embedding_matrix, embed_size=embed_size)
model.summary()
model.compile(loss="binary_crossentropy", optimizer='adam',  metrics=["accuracy"])

ckpt = ModelCheckpoint(MODEL_FILENAME, save_best_only=True, verbose=True)

model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=10, batch_size=256, callbacks=[ckpt], verbose=2)

# reference from another good scoring model in the toxic comments challenge: loss: 0.0425 - acc: 0.9844