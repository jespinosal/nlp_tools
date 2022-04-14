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
        self.valid_labels = self._get_labels(nlp_model, valid_labels)

    @abstractmethod
    def _get_labels(self, nlp_model, valid_labels):
        pass

    @abstractmethod
    def get_token_tag(self, token: spacy.tokens.token.Token):
        pass

    def process_documents(self, data):
        features = []
        if self.n_process > 1:
            with mtp.Pool(processes=self.n_process) as pool:
                features = pool.map(self.transformation, data)
        else:
            for text in tqdm(data):
                features.append(self.transformation(text))
        return features

    def transformation(self, spacy_docs):
        texts_pos_tags = []
        for doc in spacy_docs:
            text_pas_tag = [self.get_token_tag(token) for token in doc if token.tag_ in self.valid_pos_tags]
            texts_pos_tags.append(" ".join(text_pas_tag))

    def fit_transform(self, X: List[spacy.tokens.doc.Docs]):
        fitted_corpus = self.process_documents(X)
        vectorizer = TfidfVectorizer(max_features=len(fitted_corpus), analyzer='word', ngram_range=(1, 2))
        return vectorizer


class FeatureExtractionPartOfSpeech(FeatureExtractionTags):
    # Fine grained pos tag
    def _get_labels(self, nlp_model, valid_labels):
        valid_labels = valid_labels if valid_labels is None else sorted(nlp_model.get_pipe('tagger').labels)
        return valid_labels

    def get_token_tag(self, token):
        return token.tag_


class FeatureExtractionDependencyParser(FeatureExtractionTags):

    def _get_labels(self, nlp_model, valid_labels):
        valid_labels = valid_labels if valid_labels is None else sorted(nlp_model.get_pipe('parser').labels)
        return valid_labels

    def get_token_tag(self, token):
        return token.dep_


class FeatureExtractionPosTagging(FeatureExtractionTags):

    def _get_labels(self, nlp_model, valid_labels):
        pos_list = 'ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', \
                   'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'SPACE'
        valid_labels = valid_labels if valid_labels is None else pos_list
        return valid_labels

    def get_token_tag(self, token):
        return token.pos_


class FeatureExtractionNER(FeatureExtractionTags):

    def _get_labels(self, nlp_model, valid_labels):
        valid_labels = valid_labels if valid_labels is None else sorted(nlp_model.get_pipe('ner').labels)
        return valid_labels

    def get_token_tag(self, token):
        return token.ent_type_


class FeatureExtractionWordEmbeddings:
    pass


if __name__ == '__main__':
    from data_parser.preprocessing import TextPreprocessing
    pass


