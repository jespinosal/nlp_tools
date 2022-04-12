import re
import string
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import spacy
import pandas as pd


def clean_text(text):
    """

    :param text:
    :return:
    """
    text = re.sub('<.*?>', ' ', text)
    text = text.translate(str.maketrans(' ', ' ', string.punctuation))
    text = re.sub('[0-9]', ' ', text)
    text = re.sub("\n", " ", text)
    text = text.lower()
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


class TextPreprocessing:

    def __init__(self, batch_size=64, n_process=-1, nlp_model=None):
        self.nlp_model = spacy.load('en_core_web_sm') if nlp_model is None else nlp_model
        self.batch_size = batch_size
        self.n_process = n_process
        self._ps = PorterStemmer()

    def _stemming(self, word):
        return self._ps.stem(word)

    def process_documents(self, texts, config):
        texts_preprocessed = []
        spacy_docs = []
        disable_pipe = ['NER'] if config.get('ner_masking', False) else []
        for doc in tqdm(self.nlp_model.pipe(texts=texts,
                                            batch_size=self.batch_size,
                                            n_process=self.n_process,
                                            disable=disable_pipe)):
            text_word_tokens = []
            # Word token level processing
            for word_token in doc:
                text_word = self._word_token_processing(word_token, config)
                text_word_tokens.append(str(text_word))  # to parse spacy.token in case is not processed
            text_preprocessed = ' '.join(text_word_tokens)

            # String level processing
            text_preprocessed = self._string_text_processing(text_preprocessed, config)
            texts_preprocessed.append(text_preprocessed)

            # save spacy docs for future processing
            spacy_docs.append(doc)  # or add feature extraction here

        return spacy_docs, texts_preprocessed

    def _word_token_processing(self, word_token, config):  # to get postag, deptag,
        if config.get('stop_words', False):
            if word_token.text in config['stop_words']:
                return ' '

        if config.get('ner_masking', False):
            print('ent-.....')
            if word_token.ent_type_ != '':
                return str(word_token.ent_type_)

        if config.get('lemma', False):
            return word_token.lemma_

        if config.get('steam', False):
            return self._stemming(word_token.text)

    @staticmethod
    def _string_text_processing(text, config):
        """

        :param text:
        :param config:
        :return:
        """
        if config.get('delete_html', False):
            text = re.sub('<.*?>', ' ', text)
        if config.get('delete_punctuation', False):
            text = text.translate(str.maketrans(' ', ' ', string.punctuation))
        if config.get('delete_numbers', False):
            text = re.sub('[0-9]', ' ', text)
        if config.get('delete_new_line', False):
            text = re.sub("\n", " ", text)
        if config.get('lowercase', False):
            text = text.lower()
        if config.get('delete_white_spaces', False):
            text = ' '.join(text.split())
        return text


if __name__ == "__main__":

    text_path = 'data/sentiment_analysis.csv'
    df = pd.read_csv(text_path, sep=';', dtype={'label_emotion': 'str', 'partition': 'str'})[0:20000]
    texts = df.text.to_list()
    nlp_model = spacy.load('en_core_web_sm')
    batch_size = 512
    n_process = 4
    config = {'delete_html': True,
              'delete_numbers': True,
              'delete_new_line': True,
              'delete_punctuation': True,
              'delete_white_spaces': True,
              'lowercase': True,
              'ner_masking': True,
              'steam': False,
              'lemma': True,
              'stop_words': ['the', 'or', 'and', 'a', 'to', 'of', 'as']}

    text_preprocessor = TextPreprocessing(batch_size=batch_size, n_process=n_process, nlp_model=nlp_model)
    docs, texts = text_preprocessor.process_documents(texts=texts, config=config)
    import sys
    print(sys.getsizeof(docs))




