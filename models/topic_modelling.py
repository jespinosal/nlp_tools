import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from typing import List, Tuple
import matplotlib.pyplot as plt
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from sklearn.base import BaseEstimator, TransformerMixin


SEED = 100
random.seed(SEED)


class LDAModelMgr(BaseEstimator, TransformerMixin):

    def __init__(self,
                 top_words_topic_name=5
                 ):
        self.common_dictionary = None
        self.lda_model = None
        self.corpus = None
        self.top_words_topic_name = top_words_topic_name
        self.topic_names = {}
        self.coherence_per_topic = None

    @property
    def topic_names_inverse(self):
        return {topic_name: topic_id for topic_id, topic_name in self.topic_names.items()}

    def get_corpus(self, tokenized_text: List[List[str]]):
        """
         Build the text corpus using the common_dictionary, if it is None, it will be generated
        :param tokenized_corpus:
        :param common_dictionary:
        :return:
        """
        # Transform corpus in bag of word format (token_id, token_count)
        corpus = [self.common_dictionary.doc2bow(token_list) for token_list in tokenized_text]

        return corpus

    @staticmethod
    def plot_tuning(best_coherence, best_k, k_values, coherence):
        # plot LDA results for coherence score
        print('best_coherence:', best_coherence, 'best_k:', best_k)
        plt.plot(k_values, coherence)
        plt.plot(k_values, coherence, 'bo')
        plt.plot(best_k, best_coherence, 'ro')
        plt.xlabel("Num K Topics")
        plt.ylabel("Coherence score")
        plt.show()

    def set_word_dictionary(self, tokenized_text):
        self.common_dictionary = Dictionary(tokenized_text)

    def set_topic_names(self):
        topic_names = {}
        for n_topic in range(self.lda_model.num_topics):
            topic_words = [self.common_dictionary.id2token[token_id] for token_id, topic_weight in
                           self.lda_model.get_topic_terms(n_topic, topn=self.top_words_topic_name)]
            topic_names[n_topic] = " ".join(topic_words)
        return topic_names

    def fit(self, X, n_topics_range: Tuple[int] = (10, 20), n_process=1):
        """
        LDA hyper tunining to find the best LDA model
        """
        tokenized_text = X
        start_time = datetime.now()
        self.set_word_dictionary(tokenized_text)
        self.corpus = self.get_corpus(tokenized_text=tokenized_text)

        # LDA parallelization using multiprocessing CPU cores to parallelize
        k_values = list(range(n_topics_range[0], n_topics_range[1]))
        coherence = []
        model_list = []
        coherence_objects = []
        for topic_n in tqdm(k_values):
            lda_model = LdaMulticore(corpus=self.corpus,  # data_frame_corpus['corpus_tokens'].to_list()
                                     id2word=self.common_dictionary,
                                     num_topics=topic_n,
                                     workers=n_process,
                                     random_state=SEED,
                                     chunksize=100,
                                     passes=10,
                                     # alpha='auto',  #  auto-tuning alpha not implemented in multicore LDA
                                     per_word_topics=True)

            coherence_lda = CoherenceModel(model=lda_model,
                                           texts=tokenized_text,
                                           coherence='c_v')
            coherence_objects.append(coherence_lda)
            model_list.append(lda_model)
            coherence.append(coherence_lda.get_coherence())

        # get the tuning results, models and parameters
        best_model_id = np.argmax(coherence)
        self.lda_model = model_list[best_model_id]
        best_coherence = coherence[best_model_id]
        best_k = k_values[best_model_id]

        # get coherence summary per topic
        best_coherence_lda = coherence_objects[best_model_id]
        self.coherence_per_topic = best_coherence_lda.get_coherence_per_topic()

        self.plot_tuning(best_coherence, best_k, k_values, coherence)

        end_time = datetime.now()
        print(f'Duration: {end_time - start_time}')

        self.topic_names = self.set_topic_names()

        return self.lda_model

    @staticmethod
    def normalize_gamma(inference):
        return np.divide(inference, inference.sum(1).reshape(-1, 1))

    def transform(self, X: List[List[int]]):
        tokenized_text = X
        if self.common_dictionary is None:
            raise ValueError("Set a common_dictionary dict, loading a existing one or using fit method")
        elif self.lda_model is None:
            raise ValueError("Load or train a LDA model")
        else:
            corpus = self.get_corpus(tokenized_text)
        predictions, _ = self.lda_model.inference(corpus)
        predictions_normalized = self.normalize_gamma(predictions)
        return pd.DataFrame(predictions_normalized).rename(columns=self.topic_names)


if __name__ == "__main__":

    from data_parser.preprocessing import TextPreprocessing
    import pandas as pd
    import time
    import spacy

    start = time.time()
    text_path = 'data/sentiment_analysis.csv'
    df = pd.read_csv(text_path, sep=';', dtype={'label_emotion': 'str', 'partition': 'str'})[0:20000]
    texts = df.text.to_list()
    nlp_model = spacy.load('en_core_web_lg')
    batch_size = 256
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
              'stop_words': ['the', 'or', 'and', 'a', 'to', 'of', 'as', 'be', 'in']}

    text_preprocessor = TextPreprocessing(nlp_model=nlp_model,
                                          config=config,
                                          batch_size=batch_size,
                                          n_process=n_process)
    start_t = time.time()
    text_cleaned = text_preprocessor.pre_processing_texts(texts=texts)
    print('time cleaning:', time.time()-start_t)
    start_t = time.time()
    docs = text_preprocessor.process_nlp_documents(texts=text_cleaned)
    print('time spacy loading:', time.time()-start_t)
    start_t = time.time()
    texts_pos_processed = text_preprocessor.pos_processing_texts(spacy_docs=docs)
    print('time post processing:', time.time() - start_t)
    end = time.time()
    print('time:', end - start)
    _tokenized_text = [text_clean.split() for text_clean in texts_pos_processed]

    lda_model_mgr = LDAModelMgr()
    _ = lda_model_mgr.fit(tokenized_text=_tokenized_text, n_process=2, n_topics_range=(5, 10))
    predictions = lda_model_mgr.transform(tokenized_text=_tokenized_text[0:100])

