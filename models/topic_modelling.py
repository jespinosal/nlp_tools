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
                 common_dictionary: Dictionary = None,
                 lda_model: LdaMulticore = None
                 ):
        self.common_dictionary = common_dictionary
        self.lda_model = lda_model

    @staticmethod
    def set_common_dictionary(train_tokenized_text):
        """
        :param common_dictionary:
        :return:
        """
        return Dictionary(train_tokenized_text)

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

    def fit(self, tokenized_text):
        self.common_dictionary = self.set_common_dictionary(train_tokenized_text=tokenized_text)

    def transform(self, tokenized_text, n_topics_range: Tuple[int] = (10, 20), n_process=1):
        start_time = datetime.now()
        corpus = self.get_corpus(tokenized_text=tokenized_text)

        # LDA parallelization using multiprocessing CPU cores to parallelize
        k_values = list(range(n_topics_range[0], n_topics_range[1]))
        coherence = []
        model_list = []
        for topic_n in tqdm(k_values):
            lda_model = LdaMulticore(corpus=corpus,  # data_frame_corpus['corpus_tokens'].to_list()
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
            model_list.append(lda_model)
            coherence.append(coherence_lda.get_coherence())

        # get the tuning results, models and parameters
        best_model_id = np.argmax(coherence)
        self.lda_model = model_list[best_model_id]
        best_coherence = coherence[best_model_id]
        best_k = k_values[best_model_id]

        self.plot_tuning(best_coherence, best_k, k_values, coherence)

        end_time = datetime.now()
        print(f'Duration: {end_time - start_time}')
        return model_list, k_values, coherence, self.common_dictionary, self.lda_model, corpus

    def _parse_inference_matrix(self):
        pass

    def get_topic_names(self, topn=10):
        topic_names = {}
        for n_topic in range(self.lda_model.num_topics):
            topic_words = [self.common_dictionary.id2token[token_id] for token_id, topic_weight in
                           self.lda_model.get_topic_terms(n_topic, topn=topn)]
            topic_names[n_topic] = " ".join(topic_words)
        return topic_names

    def predict(self, tokenized_text: List[List[int]], topn=3):
        if self.common_dictionary is None:
            raise ValueError("Set a common_dictionary dict, loading a existing one or using fit method")
        elif self.lda_model is None:
            raise ValueError("Load or train a LDA model")
        else:
            corpus = self.get_corpus(tokenized_text)
        predictions, _ = self.lda_model.inference(corpus)
        return pd.DataFrame(predictions).rename(columns=self.get_topic_names(topn=topn))


if __name__ == "__main__":

    from data_parser.preprocessing import TextPreprocessing
    import pandas as pd
    import time
    import spacy

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
    _tokenized_text = [text_clean.split() for text_clean in texts_pos_processed]

    lda_model_mgr = LDAModelMgr()
    lda_model_mgr.fit(tokenized_text=_tokenized_text)
    _ = lda_model_mgr.transform(tokenized_text=_tokenized_text, n_process=2, n_topics_range=(5, 10))
    predictions = lda_model_mgr.predict(tokenized_text=_tokenized_text[0:100])

