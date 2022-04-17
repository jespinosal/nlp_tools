import re
from typing import List
import string
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import spacy
import multiprocessing as mtp


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

    def __init__(self, nlp_model: spacy.language.Language,
                       config: dict,
                       batch_size=1024,
                       n_process=2):
        self.nlp_model = spacy.load('en_core_web_sm') if nlp_model is None else nlp_model
        self.batch_size = batch_size
        self.n_process = n_process
        self._ps = PorterStemmer()
        self.text_preproceded = False
        self.text_spacy_processed = False
        self.config = config

    @staticmethod
    def split_abbreviation(text):
        """
        Process common abbreviations on isolate tokens
        :param text:
        :return:
        """
        text = re.sub(r"\'s", " \'s", text)
        text = re.sub(r"\'ve", " \'ve", text)
        text = re.sub(r"n\'t", " n\'t", text)
        text = re.sub(r"\'re", " \'re", text)
        text = re.sub(r"\'d", " \'d", text)
        text = re.sub(r"\'ll", " \'ll", text)
        return text

    @staticmethod
    def split_punctuation(text):
        text = re.sub(r",", " , ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\(", " \( ", text)
        text = re.sub(r"\)", " \) ", text)
        text = re.sub(r"\?", " \? ", text)
        return text

    def _stemming(self, word):
        return self._ps.stem(word)

    def process_nlp_documents(self, texts: List[str]):
        spacy_docs = []
        disable_pipe = ['NER'] if self.config.get('ner_masking', False) else []
        for doc in tqdm(self.nlp_model.pipe(texts=texts,
                                            batch_size=self.batch_size,
                                            n_process=self.n_process,
                                            disable=disable_pipe)):
            spacy_docs.append(doc)

        return spacy_docs

    def _pos_processing_text(self, doc):
        text_word_tokens = []

        # Word token level processing
        for word_token in doc:
            text_word = self._word_token_processing(word_token)
            text_word_tokens.append(str(text_word))  # to parse spacy.token in case is not processed
        text_preprocessed = ' '.join(text_word_tokens)

        # String level processing
        text_preprocessed = self._string_text_pos_processing(text_preprocessed)

        return text_preprocessed

    def pos_processing_texts(self, spacy_docs):
        texts_pos_processed = []
        if self.n_process > 1:
            with mtp.Pool(processes=self.n_process) as pool:
                texts_pos_processed = pool.map(self._pos_processing_text, spacy_docs)
        else:
            for doc in tqdm(spacy_docs):
                texts_pos_processed.append(self._pos_processing_text(doc))

        return texts_pos_processed

    def pre_processing_texts(self, texts):
        texts_pre_processed = []
        if self.n_process > 1:
            with mtp.Pool(processes=self.n_process) as pool:
                texts_pre_processed = pool.map(self._string_text_pre_processing, texts)
        else:
            for text in tqdm(texts):
                texts_pre_processed.append(self._string_text_pre_processing(text))

        return texts_pre_processed

    def _word_token_processing(self, word_token: spacy.tokens.token):
        """
        Implements transformation at token level
        :param word_token:
        :param config:
        :return:
        """
        if self.config.get('stop_words', False):
            if word_token.text in self.config['stop_words']:
                return ' '

        if self.config.get('ner_masking', False):
            if word_token.ent_type_ != '':
                return str(word_token.ent_type_)

        if self.config.get('lemma', False):
            return word_token.lemma_

        if self.config.get('steam', False):
            return self._stemming(word_token.text)

    def _string_text_pos_processing(self, text):
        """

        :param text:
        :param config:
        :return:
        """
        if self.config.get('delete_punctuation', False):
            text = text.translate(str.maketrans(' ', ' ', string.punctuation))
        if self.config.get('delete_numbers', False):
            text = re.sub('[0-9]', ' ', text)
        if self.config.get('lowercase', False):
            text = text.lower()
        if self.config.get('split_punctuation'):
            text = self.split_punctuation(text)
        if self.config.get('split_abbreviation'):
            text = self.split_abbreviation(text)
        if self.config.get('delete_white_spaces', False):
            text = ' '.join(text.split())
        return text

    def _string_text_pre_processing(self, text):
        if self.config.get('delete_new_line', False):
            text = re.sub("\n", " ", text)
        if self.config.get('delete_html', False):
            text = re.sub('<.*?>', ' ', text)
        if self.config.get('delete_white_spaces', False):
            text = ' '.join(text.split())
        return text


if __name__ == "__main__":

    import pandas as pd
    import time
    start = time.time()
    text_path = 'data/sentiment_analysis.csv'
    df = pd.read_csv(text_path, sep=';', dtype={'label_emotion': 'str', 'partition': 'str'})[0:50000]
    texts = df.text.to_list()
    nlp_model = spacy.load('en_core_web_sm')
    batch_size = 512
    n_process = 1
    config = {'delete_html': True,
              'delete_numbers': True,
              'delete_new_line': True,
              'delete_punctuation': True,
              'delete_white_spaces': True,
              'lowercase': True,
              'ner_masking': True,
              'steam': False,
              'lemma': True,
              'stop_words': ['the', 'or', 'and', 'a', 'to', 'of', 'as'],
              'split_punctuation': True,
              'split_abbreviations': True}

    text_preprocessor = TextPreprocessing(nlp_model=nlp_model,
                                          config=config,
                                          batch_size=batch_size,
                                          n_process=n_process)
    text_cleaned = text_preprocessor.pre_processing_texts(texts=texts)
    docs = text_preprocessor.process_nlp_documents(texts=text_cleaned)
    texts_pos_processed = text_preprocessor.pos_processing_texts(spacy_docs=docs)
    import sys
    print(sys.getsizeof(docs))
    end = time.time()
    print('time:', end-start)




