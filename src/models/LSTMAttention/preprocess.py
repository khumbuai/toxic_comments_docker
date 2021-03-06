import re
from nltk import word_tokenize


def preprocess(comment):
    comment = comment.replace('&', ' and ')
    comment = comment.replace('0', ' zero ')
    comment = comment.replace('1', ' one ')
    comment = comment.replace('2', ' two ')
    comment = comment.replace('3', ' three ')
    comment = comment.replace('4', ' four ')
    comment = comment.replace('5', ' five ')
    comment = comment.replace('6', ' six ')
    comment = comment.replace('7', ' seven ')
    comment = comment.replace('8', ' eight ')
    comment = comment.replace('9', ' nine ')
    comment = comment.replace('\'ve', ' have ')
    comment = comment.replace('\'d', ' would ')
    comment = comment.replace('\'m', ' am ')
    comment = comment.replace('n\'t', ' not ')
    comment = comment.replace('\'s', ' is ')
    comment = comment.replace('\'r', ' are ')
    comment = re.sub(r"\\", "", comment)
    comment = word_tokenize(comment)
    comment = " ".join(word for word in comment)
    return comment.strip()
