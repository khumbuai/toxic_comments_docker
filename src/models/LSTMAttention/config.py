maxlen=150
max_features=100000
embed_size=300

EMBEDDING_FILE = "../../assets/crawl-300d-2M.vec"
TRAIN_FILENAME = "../../assets/train.csv"
TOKENIZER_FILENAME = './tokenizer.p'
MODEL_FILENAME = './model.hdf5'

categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
