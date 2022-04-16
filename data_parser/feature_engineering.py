from abc import ABC, abstractmethod
import multiprocessing as mtp
from tqdm import tqdm
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List


class FeatureExtractionTags(ABC, BaseEstimator, TransformerMixin):

    def __init__(self,
                 nlp_model: spacy.language.Language,
                 valid_labels: list = None,
                 n_process: int = 1,
                 ):
        self.n_process = n_process
        self.fitted_corpus = None
        self.default_token = 'EMPTY'
        self.valid_labels = self._get_labels(nlp_model, valid_labels) + [self.default_token]
        self.consider_intermediate_tokens = True
        self.return_dataset = False

    @abstractmethod
    def _get_labels(self, nlp_model, valid_labels):
        pass

    @abstractmethod
    def get_token_tag(self, token: spacy.tokens.token.Token):
        pass

    def process_documents(self, spacy_docs):
        features = []
        if self.n_process > 1:
            with mtp.Pool(processes=self.n_process) as pool:
                features = pool.map(self.transformation, spacy_docs)
        else:
            for doc in tqdm(spacy_docs):
                features.append(self.transformation(doc))
        return features

    def transformation(self, doc):
        """
        Applies the transformation to get the wanted features at token level, concatenating all the found features
        in a text string.
        When considering consider_intermediate_tokens will return a string with the same size of original text.
        If consider_intermediate_tokens is False will return a text including only the tokens with the wanted
        valid_labels.
        The above cases will apply only when valid_labels has not the default value and mainly for NER uses cases.
        Examples for NER MODULE with COUNTRY AND MONEY CATEGORIES:
        If consider_intermediate_tokens = True
        text = "Hello in Canada we spent more than 10$ per day at home"
        >> EMPTY EMPTY COUNTRY EMPTY EMPTY EMPTY EMPTY MONEY EMPTY EMPTY EMPTY
        If consider_intermediate_tokens = False
        >> COUNTRY MONEY

        :param spacy_docs:
        :return:
        """
        if self.consider_intermediate_tokens:
            text_tags = [self.get_token_tag(token) if self.get_token_tag(token) in self.valid_labels
                         else self.default_token for token in doc]
        else:
            text_tags = [self.get_token_tag(token) for token in doc if self.get_token_tag(token)
                         in self.valid_labels]
        return " ".join(text_tags)

    def fit_transform(self, X: List[spacy.tokens.doc.Doc]):
        fitted_corpus = self.process_documents(X)
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
        sparse_vectors = vectorizer.fit_transform(fitted_corpus)
        if self.return_dataset:
            dense_vector = sparse_vectors.todense().tolist()
            return pd.DataFrame(dense_vector, columns=vectorizer.get_feature_names())
        else:
            sparse_vectors


class FeatureExtractionPartOfSpeech(FeatureExtractionTags):
    # Fine grained pos tag
    def _get_labels(self, nlp_model, valid_labels):
        valid_labels = valid_labels if valid_labels is not None else sorted(nlp_model.get_pipe('tagger').labels)
        return valid_labels

    def get_token_tag(self, token):
        return token.tag_


class FeatureExtractionDependencyParser(FeatureExtractionTags):

    def _get_labels(self, nlp_model, valid_labels):
        valid_labels = valid_labels if valid_labels is not None else sorted(nlp_model.get_pipe('parser').labels)
        return valid_labels

    def get_token_tag(self, token):
        return token.dep_


class FeatureExtractionPosTagging(FeatureExtractionTags):

    def _get_labels(self, nlp_model, valid_labels):
        pos_list = 'ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', \
                   'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'SPACE'
        valid_labels = valid_labels if valid_labels is not None else pos_list
        return valid_labels

    def get_token_tag(self, token):
        return token.pos_


class FeatureExtractionNER(FeatureExtractionTags):

    def _get_labels(self, nlp_model, valid_labels):
        valid_labels = valid_labels if valid_labels is not None else sorted(nlp_model.get_pipe('ner').labels)
        return valid_labels

    def get_token_tag(self, token):
        return token.ent_type_


class FeatureExtractionWordEmbeddings:
    pass


if __name__ == '__main__':

    from data_parser.preprocessing import TextPreprocessing
    import pandas as pd
    import time

    start = time.time()
    text_path = 'data/sentiment_analysis.csv'
    df = pd.read_csv(text_path, sep=';', dtype={'label_emotion': 'str', 'partition': 'str'})[0:1000]
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
              'stop_words': ['the', 'or', 'and', 'a', 'to', 'of', 'as']}

    text_preprocessor = TextPreprocessing(nlp_model=nlp_model,
                                          config=config,
                                          batch_size=batch_size,
                                          n_process=n_process)
    text_cleaned = text_preprocessor.pre_processing_texts(texts=texts)
    docs = text_preprocessor.process_nlp_documents(texts=text_cleaned)
    texts_pos_processed = text_preprocessor.pos_processing_texts(spacy_docs=docs)
    end = time.time()
    print('time:', end - start)


    ### FEATURES
    featurizer_ner = FeatureExtractionNER(nlp_model=nlp_model)
    text_document = featurizer_ner.transformation(doc=docs[0])
    features_ner = featurizer_ner.fit_transform(X=docs)


    # Return features on DF form for EDA get_features()
    # todo test sparce and dense output and add wemb



