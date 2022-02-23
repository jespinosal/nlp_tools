import re
import string
from nltk.tokenize import word_tokenize


def clean_text(text):
    """

    :param text:
    :return:
    """
    # will replace the html characters with " "
    text = re.sub('<.*?>', ' ', text)
    # To remove the punctuations
    text = text.translate(str.maketrans(' ', ' ', string.punctuation))
    # will consider only alphabets and numerics
    text = re.sub('[^a-zA-Z]', ' ', text)
    # will replace newline with space
    text = re.sub("\n", " ", text)
    # will convert to lower case
    text = text.lower()
    # will split and join the words
    text = ' '.join(text.split())
    return text


def stopwords(input_text, stop_words):
    """

    :param input_text:
    :param stop_words:
    :return:
    """
    word_tokens = word_tokenize(input_text)
    output = []
    for w in word_tokens:
        if w not in stop_words:
            output.append(w)
    text = ' '.join(output)
    return text
