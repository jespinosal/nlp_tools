from abc import ABC, abstractmethod
import multiprocessing as mtp
from tqdm import tqdm
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List
from models.topic_modelling import LDAModelMgr
import pandas as pd


class FeatureExtractionBOWBase:
    def __init__(self,
                 max_features=50000
                 ):
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=max_features)
        self.feature_names = None

    def _get_feature_names(self):
        if self.vectorizer is not None:
            self.feature_names = self.vectorizer.get_feature_names_out()
        else:
            Exception("Vectorizer should be fitted first")

    # The private methods has to purpose 1) Build fit_transform and 2) Support BOW
    def _fit(self, fitted_corpus):
        self.vectorizer.fit(fitted_corpus)
        self._get_feature_names()
        return None

    def _transform(self, fitted_corpus):
        sparse_vectors = self.vectorizer.fit_transform(fitted_corpus)
        dense_vector = sparse_vectors.todense().tolist()
        return pd.DataFrame(dense_vector, columns=self.feature_names)


class FeatureExtractionTags(ABC, BaseEstimator, TransformerMixin, FeatureExtractionBOWBase):

    def __init__(self,
                 nlp_model: spacy.language.Language,
                 valid_labels: list = None,
                 n_process: int = 1,
                 max_features=50000
                 ):
        self.n_process = n_process
        self.fitted_corpus = None
        self.default_token = 'EMPTY'
        self.valid_labels = self._get_labels(nlp_model, valid_labels) + [self.default_token]
        self.consider_intermediate_tokens = True
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=max_features)
        self.feature_names = None  # todo add super class init

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

    def fit(self, X: List[spacy.tokens.doc.Doc]):
        fitted_corpus = self.process_documents(X)
        self._fit(fitted_corpus)
        return None

    def transform(self, X: List[spacy.tokens.doc.Doc]):
        fitted_corpus = self.process_documents(X)
        self._transform(fitted_corpus)
        return self._transform(fitted_corpus)

    def fit_transform(self, X: List[spacy.tokens.doc.Doc]):
        fitted_corpus = self.process_documents(X)
        self._fit(fitted_corpus)
        return self._transform(fitted_corpus)


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
        pos_list = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', \
                   'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'SPACE']
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


class FeatureExtractionWordEmbeddings(BaseEstimator, TransformerMixin):
    """
    Remember for small-medioum models embeddings will be hash fucntions.
    If you want word embeddings such as Glove use large versions.
    """
    def fit_transform(self, X: List[spacy.tokens.doc.Doc]):
        """
        Get average vector of fasttext spacy documents
        :return:
        """
        return pd.DataFrame([doc.vector for doc in X])


class FeatureExtractionBOW(FeatureExtractionBOWBase):

    def fit(self, X: List[str]):
        self._fit(X)
        return None

    def transform(self, X: List[str]):
        return self._transform(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class FeatureExtractionTopics(BaseEstimator, TransformerMixin):

    def __init__(self, n_process=2, min_topics=10, max_topics=30):
        self.lda_model_mgr = None
        self.n_process = n_process
        self.min_topics = min_topics
        self.max_topics = max_topics

    @staticmethod
    def tokenize_text(texts):
        return [text.split() for text in texts]

    def fit(self, X: List[List[str]]):
        tokenized_text = self.tokenize_text(X)
        self.lda_model_mgr = LDAModelMgr()
        self.lda_model_mgr.fit(tokenized_text=tokenized_text)
        _ = self.lda_model_mgr.transform(tokenized_text=tokenized_text,
                                         n_process=self.n_process,
                                         n_topics_range=(self.min_topics, self.max_topics))

    def transform(self, X: List[List[str]]):
        tokenized_text = self.tokenize_text(X)
        return self.lda_model_mgr.predict(tokenized_text=tokenized_text)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.transform(X)


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
    features_ner = featurizer_ner.fit_transform(X=docs)

    featurizer_pos = FeatureExtractionPartOfSpeech(nlp_model=nlp_model)
    features_pos = featurizer_pos.fit_transform(X=docs)

    featurizer_dep_par = FeatureExtractionDependencyParser(nlp_model=nlp_model)
    features_dep_par = featurizer_dep_par.fit_transform(X=docs)

    featurizer_pos_tags = FeatureExtractionPosTagging(nlp_model=nlp_model)
    features_pos_tags = featurizer_pos_tags.fit_transform(X=docs)

    featurizer_topics = FeatureExtractionTopics(n_process=2,
                                                min_topics=10,
                                                max_topics=20)
    features_topics = featurizer_topics.fit_transform(texts_pos_processed)

    featurizer_word_emb = FeatureExtractionWordEmbeddings()
    features_word_emb = featurizer_word_emb.fit_transform(X=docs)

    featurizer_bow = FeatureExtractionBOW(max_features=100)
    features_pos_tags = featurizer_bow.fit_transform(X=text_cleaned)




