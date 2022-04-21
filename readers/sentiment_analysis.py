"""
This script will read data from different sources to build a standard data set to perform three type
of task:

- Emotion classification (set emotions Paul Ekman : happiness, sadness, anger, fear, surprise, and disgust)
- Polarity (-1, 0 , 1)
- Valence/Arousal/Dominance

The data sources used involves:
- GoEmotion
- ISEAR
- GoodNewsEveryone
- Emobank
- IEmocap
- Dialogues todo now (13k) [6 EMOTIONS]
- EECS todo (1K) students dataset topics comments at university [POS-NEG-NEU]
- Emotion Cause  todo now (1594 + 820 = 2.4K) [6 EMOTIONS]
- AffectiveText todo 1k-news ids -> anger disgust fear joy sadness surprise. [6 EMOTIONS / POS-NEG-NEU]



Emotion modeling (from: A survey of state-of-the-art approaches for emotion
recognition in text):

Psychology research has distinguished three major approaches for emotion modeling [60,62].
Table 1 summarizes the three dominant emotion modeling approaches:

– Categorical approach This approach is based on the idea that there exist a small number
of emotions that are basic and universally recognized [62]. The most commonly used
model in emotion recognition research is that of Paul Ekman [44], which involves six
basic emotions: happiness, sadness, anger, fear, surprise, and disgust.

– Dimensional approach This approach is based on the idea that emotional states are not
independent but are related to each other in a systematic manner [62]. This approach
covers emotion variability in three dimensions [20,78]:
– Valence: This dimension refers to how positive or negative an emotion is [62].
– Arousal: This dimension refers to how excited or apathetic an emotion is [62].
– Power: This dimension refers to the degree of power [62].

- Appraisal-based approach This approach can be considered as an extension of the dimensional approach.
It uses componential models of emotion based on appraisal theory [132],
which states that a person can experience an emotion if it is extracted via an evaluation
of events and that the result is based on a person’s experience, goals, and opportunities
for action. Here, emotions are viewed through changes in all significant components,
including cognition, physiology, motivation, motor, reactions, feelings, and expressions


In the categorical approach, the emotional states are limited to a fixed number of discrete
categories, and it may be difficult to address a complex emotional state or mixed emotions
[172]. However, these types of emotions can be well addressed in the dimensional approach,
although the reduction in the emotion space to three dimensions is extreme and may result in
information loss. Furthermore, not all basic emotions fit well in the dimensional space, some
become indistinguishable, and some may lie outside the space. Regarding the advantage
of componential models, they focus on the variability of different emotional states due to
different types of appraisal patterns
"""

import pandas as pd
import numpy as np
import json
import os
import re
import string
from collections import Counter


class DataPartitions:
    TRAIN = 'train'
    TEST = 'test'
    DEV = 'dev'

    @classmethod
    def get_partition(cls, path):
        if cls.TRAIN in path:
            return cls.TRAIN
        elif cls.TEST in path:
            return cls.TEST
        else:
            return cls.DEV


class PolarityClasses:
    POSITIVE = 'positive'
    NEGATIVE = 'negative'
    NEUTRAL = 'neutral'


class ConstantsSentimentAnalysis:
    LABEL_POLARITY = 'label_polarity'
    LABEL_EMOTION = 'label_emotion'
    LABEL_EKMAN = 'label_ekman'
    TEXT = 'text'
    SOURCE = 'source'
    PARTITION = 'partition'


class ConstantsSentimentDatasetsNames:
    EMOCAP = 'EMOCAP'
    ISEAR = 'ISEAR'
    GNE = 'GNE'
    GO_EMOTIONS = 'GE'
    EMO_BANK = 'EMOBANK'
    EECS = 'EECS'  # todo
    AFFECTIVE_TEXT = 'AFFECTIVE_TEXT'  # todo
    DIALOGUES = 'DIALOGUES'
    EMO_STIMULUS = 'EMO_STIMULUS'


# map sources: https://github.com/google-research/google-research/blob/master/goemotions/data/sentiment_dict.json
# Tree map strategy : https://www.academia.edu/20456607/Emotion_knowledge_Further_exploration_of_a_prototype_approach
# https://en.wikipedia.org/wiki/Emotion_classification

extended_ambiguous = ['neutral', 'unknown']
extended_positive = ['positive', 'positive_surprise', 'happiness', 'positive_anticipation_including_optimism',
                     'trust', 'love_including_like', 'happy']
extended_negative = ['negative', 'negative_surprise', 'frustration', 'guilt', 'shame',
                     'negative_anticipation_including_pessimism', 'sad']

target_polarity_map = {
    "positive": ["amusement", "excitement", "joy", "love", "desire", "optimism", "caring", "pride", "admiration",
                 "gratitude", "relief", "approval"] + extended_positive,
    "negative": ["fear", "nervousness", "remorse", "embarrassment", "disappointment", "sadness", "grief", "disgust",
                 "anger", "annoyance", "disapproval"] + extended_negative,
    "ambiguous": ["realization", "surprise", "curiosity", "confusion"] + extended_ambiguous
}

arousal_threshold = {'positive': 0.5, 'negative': 0.49}

extended_joy = ['happy', 'love_including_like', 'positive_anticipation_including_optimism',
                'positive_surprise', 'happiness', 'trust']  # ENGAGE/CONNECT
extended_sadness = ['sad', 'negative_anticipation_including_pessimism', 'shame', 'guilt']  # RESIGNATION, FEEL ASHAMED
extended_no_emotion = ['neutral']
extended_unknown = ['unknown', 'negative', 'positive']
extended_disgust = ['negative_surprise']  # DISLIKE/AVERSION
#  extended_fear = []
extended_anger = ['frustration']  # FRUSTRATION
emotions_mask_ekman = {
    "anger": ["anger", "annoyance", "disapproval"] + extended_anger,
    "disgust": ["disgust"] + extended_disgust,
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism", "relief", "pride",
            "admiration", "desire", "caring"] + extended_joy,
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"] + extended_sadness,
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "neutral": extended_no_emotion,
    "unknown": extended_unknown
}
emotions_mask_ekman = {emotion_group: set(emotions) for emotion_group, emotions in emotions_mask_ekman.items()}


# using set type for efficient search when using the dict


def reader_dataset_go_emotions(sentiment_analysis_data_path) -> pd.DataFrame:
    """
    The go_emotions dataset include 211225 samples with 37 columns
    Columns:
    ['text', 'id', 'author', 'subreddit', 'link_id', 'parent_id',
       'created_utc', 'rater_id', 'example_very_unclear', 'admiration',
       'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
       'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
       'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy',
       'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
       'remorse', 'sadness', 'surprise', 'neutral']

    Original source contains 211225 samples in 3 partitions
    wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv
    wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv
    wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv
    Filtered source contains 54260 samples in 3 partitions (s includes examples where there is agreement between at
    least 2 raters)
    wget https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/train.tsv
    wget https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/test.tsv
    wget https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/dev.tsv
    :return:
    """
    paths = [os.path.join(sentiment_analysis_data_path, 'go_emotions_test.tsv'),
             os.path.join(sentiment_analysis_data_path, 'go_emotions_train.tsv'),
             os.path.join(sentiment_analysis_data_path, 'go_emotions_dev.tsv')]

    emotions = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
                'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
                'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
                'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']

    df_list = []

    for path in paths:
        df = pd.read_csv(path, delimiter='\t', quoting=3, header=None,
                         names=[ConstantsSentimentAnalysis.TEXT,
                                ConstantsSentimentAnalysis.LABEL_EMOTION, "annotator_id"])
        df[ConstantsSentimentAnalysis.PARTITION] = DataPartitions.get_partition(path=path)
        df_list.append(df)

    df_go_emotions = pd.concat(df_list)
    df_go_emotions[ConstantsSentimentAnalysis.LABEL_EMOTION] = df_go_emotions[
        ConstantsSentimentAnalysis.LABEL_EMOTION].apply(
        lambda x: [emotions[int(annotator_id)] for annotator_id in x.split(',')])
    df_go_emotions[ConstantsSentimentAnalysis.LABEL_POLARITY] = None
    df_go_emotions[ConstantsSentimentAnalysis.SOURCE] = ConstantsSentimentDatasetsNames.GO_EMOTIONS
    del df_go_emotions["annotator_id"]

    return df_go_emotions


def reader_dataset_emocap(sentiment_analysis_data_path) -> pd.DataFrame:
    """
    Dataset reader for emocap data
    The emocap dataset is splited in three files with the follow columns:
    ['Dialogue_ID', 'Utterance_Id', 'Utterance', 'Emotion', 'Valence', 'Activation', 'Dominance']
    Dev include  805 lines, Test include 2021 lines ,Train include 7213 lines fot a total of 10039 lines
    :param path:
    :return:
    """
    paths = [os.path.join(sentiment_analysis_data_path, 'emocap_dev.csv'),
             os.path.join(sentiment_analysis_data_path, 'emocap_test.csv'),
             os.path.join(sentiment_analysis_data_path, 'emocap_train.csv')]

    emotion_map = {
        "ang": "anger",
        "dis": "disgust",
        "exc": "excitement",
        "fea": "fear",
        "fru": "frustration",
        "hap": "happiness",
        "neu": "neutral",
        "oth": "other",
        "sad": "sadness",
        "sur": "surprise",
        "xxx": "unknown",
    }

    SWDA_DA_DESCRIPTION = {
        "+": "Unknown",
        "x": "Unknown",
        "ny": "Unknown",
        "sv_fx": "Unknown",
        "qy_qr": "Unknown",
        "ba_fe": "Unknown",
        "%": "Uninterpretble",
        "nn": "No Answers"
    }

    text_column = 'Utterance'
    label_emotion_column = 'Emotion'
    invalid_emotions = ['other']
    df_list = []
    for path in paths:
        df = pd.read_csv(path)
        df[ConstantsSentimentAnalysis.PARTITION] = DataPartitions.get_partition(path=path)
        df_list.append(df)

    df_emocap = pd.concat(df_list)
    df_emocap[ConstantsSentimentAnalysis.LABEL_EMOTION] = df_emocap[label_emotion_column].replace(emotion_map)
    df_emocap = df_emocap.rename(columns={text_column: ConstantsSentimentAnalysis.TEXT})
    df_emocap = df_emocap[~df_emocap[ConstantsSentimentAnalysis.LABEL_EMOTION].isin(invalid_emotions)]
    df_emocap[ConstantsSentimentAnalysis.LABEL_POLARITY] = None
    df_emocap['source'] = ConstantsSentimentDatasetsNames.EMOCAP
    df_emocap[ConstantsSentimentAnalysis.LABEL_EMOTION] = df_emocap[
        ConstantsSentimentAnalysis.LABEL_EMOTION].apply(lambda x: [x])

    return df_emocap[[ConstantsSentimentAnalysis.TEXT, ConstantsSentimentAnalysis.LABEL_EMOTION,
                      ConstantsSentimentAnalysis.LABEL_POLARITY, ConstantsSentimentAnalysis.SOURCE]]


def reader_dataset_good_news_everyone(sentiment_analysis_data_path) -> pd.DataFrame:
    """
    Read list of objects from a JSON lines file.
    5000 lines, rich annotations from 5 annotators
    """

    def get_dominant_categories(list_categories, threshold=2):
        """
        Filter the results with high frequency according the annotators intersection.
        The method will include examples where there is agreement between at least "threshold" raters
        :param list_categories:
        :param threshold:
        :return:
        """
        valid_categories = []
        category_frequencies = Counter(list_categories)
        for cat, freq in category_frequencies.items():
            if freq >= threshold:
                valid_categories.append(cat)
        return valid_categories

    input_path = os.path.join(sentiment_analysis_data_path, 'gne-release-v1.0.jsonl')
    data = []
    text_field = 'headline'
    label_field = 'gold'
    intensity_field = 'intensity'
    invalid_intensities = [None, 'low', 'weak']
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_line = json.loads(line.rstrip('\n|\r'))
            data.append({ConstantsSentimentAnalysis.TEXT: json_line[text_field],
                         'dominant_emotion': json_line['annotations']['dominant_emotion'][
                             label_field],
                         ConstantsSentimentAnalysis.LABEL_EMOTION: get_dominant_categories(
                             list_categories=json_line['annotations']['dominant_emotion']['raw'],
                             threshold=2),
                         intensity_field: json_line['annotations']['intensity'][label_field]})

    df_gne = pd.DataFrame(data)
    df_gne = df_gne[~df_gne[intensity_field].isin(invalid_intensities)]
    df_gne[ConstantsSentimentAnalysis.LABEL_POLARITY] = None
    df_gne['source'] = ConstantsSentimentDatasetsNames.GNE
    df_gne = df_gne[
        df_gne[ConstantsSentimentAnalysis.LABEL_EMOTION].apply(lambda x: len(x) >= 1)]  # --> 3307 samples

    return df_gne[[ConstantsSentimentAnalysis.TEXT, ConstantsSentimentAnalysis.LABEL_EMOTION,
                   ConstantsSentimentAnalysis.LABEL_POLARITY, ConstantsSentimentAnalysis.SOURCE]]


def reader_dataset_isear(sentiment_analysis_data_path) -> pd.DataFrame:
    """
    Dataset reader for emo ISEAR
    The issear have 7426 usefull lines, the dataset include the follow columns:
        columns = ['ID', 'CITY', 'COUN', 'SUBJ', 'SEX', 'AGE', 'RELI', 'PRAC', 'FOCC', 'MOCC', 'FIEL', 'EMOT', 'WHEN',
               'LONG', 'INTS',
               'ERGO', 'TROPHO', 'TEMPER', 'EXPRES', 'MOVE', 'EXP1', 'EXP2', 'EXP10', 'PARAL', 'CON', 'EXPC', 'PLEA',
               'PLAN', 'FAIR',
               'CAUS', 'COPING', 'MORL', 'SELF', 'RELA', 'VERBAL', 'NEUTRO', 'Field1', 'Field3', 'Field2', 'MYKEY',
               'SIT', 'STATE']
    :param path:
    :return:
    """

    isear_path = os.path.join(sentiment_analysis_data_path, 'isear.csv')
    text_column = 'SIT'
    label_emotion_column = 'EMOT'
    nan_token = '[ No response.]'  # 77 tokens
    emotion_map = {1: 'joy',
                   2: 'fear',
                   3: 'anger',
                   4: 'sadness',
                   5: 'disgust',
                   6: 'shame',
                   7: 'guilt'}

    df_isear = pd.read_csv(isear_path, sep="|", header='infer',
                           error_bad_lines=False)  # skiprows -> 163 lines -> 2%

    df_isear[ConstantsSentimentAnalysis.LABEL_EMOTION] = df_isear[label_emotion_column].replace(emotion_map)
    df_isear[ConstantsSentimentAnalysis.LABEL_POLARITY] = None
    df_isear = df_isear.rename(columns={text_column: ConstantsSentimentAnalysis.TEXT})
    df_isear = df_isear[df_isear[ConstantsSentimentAnalysis.TEXT] != nan_token]
    df_isear['source'] = ConstantsSentimentDatasetsNames.ISEAR
    df_isear[ConstantsSentimentAnalysis.LABEL_EMOTION] = df_isear[
        ConstantsSentimentAnalysis.LABEL_EMOTION].apply(lambda x: [x])

    return df_isear[[ConstantsSentimentAnalysis.TEXT, ConstantsSentimentAnalysis.LABEL_EMOTION,
                     ConstantsSentimentAnalysis.LABEL_POLARITY, ConstantsSentimentAnalysis.SOURCE]]


def reader_dataset_emo_bank(sentiment_analysis_data_path) -> pd.DataFrame:
    """
    Dataset reader for emo bank data
    Authors demostrated arousal, valence and dominance could mapped into the Ekmans 6 Basic emotions, where only the
    test and train emotions was mapped.
    https://github.com/JULIELab/EmoBank/issues/1 --> http://web.eecs.umich.edu/~mihalcea/affectivetext/
    :return:
    """

    def valence_to_polarity(valence, pos_threshold, neg_threshold):
        if valence > pos_threshold:
            return PolarityClasses.POSITIVE
        elif valence < neg_threshold:
            return PolarityClasses.NEGATIVE
        else:
            return PolarityClasses.NEUTRAL

    valid_categories = ['SemEval', 'blog']
    valence_column = 'V'
    df_emobank = pd.read_csv(os.path.join(sentiment_analysis_data_path, 'emobank.csv'), index_col=0)
    df_meta = pd.read_csv(os.path.join(sentiment_analysis_data_path, 'meta.tsv'), sep='\t', index_col=0)
    df_emobank = df_emobank.join(df_meta, how='inner')
    df_emobank[ConstantsSentimentAnalysis.SOURCE] = ConstantsSentimentDatasetsNames.EMO_BANK
    valence_mean = np.mean(df_emobank[valence_column])
    valence_sdt = np.std(df_emobank[valence_column])
    pos_threshold = valence_mean + valence_sdt
    neg_threshold = valence_mean - valence_sdt
    df_emobank[ConstantsSentimentAnalysis.LABEL_POLARITY] = df_emobank[valence_column].apply(
        lambda valence: valence_to_polarity(valence, pos_threshold, neg_threshold))
    df_emobank = df_emobank[df_emobank.category.isin(valid_categories)]
    df_emobank[ConstantsSentimentAnalysis.LABEL_EMOTION] = df_emobank[
        ConstantsSentimentAnalysis.LABEL_POLARITY].apply(
        lambda x: [x])

    return df_emobank[[ConstantsSentimentAnalysis.TEXT, ConstantsSentimentAnalysis.LABEL_EMOTION,
                       ConstantsSentimentAnalysis.LABEL_POLARITY, ConstantsSentimentAnalysis.SOURCE]]


def reader_dataset_emotion_stimulus(sentiment_analysis_data_path):
    """
    Reader for emo stimulus dataset, with 2.4k samples with the emotions 'happy', 'sad', 'surprise', 'disgust',
    'anger', 'fear', 'shame'. The data comes from FrameNet labeling multi domain comments
    :return:
    """

    def get_label(text_labeled):
        emotions = []
        for emotion in cause_sentiments_init_tokens:
            if emotion in text_labeled:
                emotions.append(text_preprocessing_basic(emotion))
        return emotions

    def clean_annotations(text_labeled):
        for token in annotation_tokens:
            if token in text_labeled:
                text_labeled = text_labeled.replace(token, ' ')

        return text_labeled

    labeled_text = 'labeled_text'
    emotions_ = ['happy', 'sad', 'surprise', 'disgust', 'anger', 'fear', 'shame']
    cause_sentiments_init_tokens = [f'<{emotion}>' for emotion in emotions_]
    cause_sentiments_end_tokens = [token.replace('<', '<\\') for token in cause_sentiments_init_tokens]
    cause_tokens = ['<cause>', '<\\cause>']  # replace using this dict
    annotation_tokens = cause_sentiments_init_tokens + cause_sentiments_end_tokens + cause_tokens
    df_cause = pd.read_table(os.path.join(sentiment_analysis_data_path, 'emotion_stimulus_cause.txt'),
                             delim_whitespace=False, header=0, names=[labeled_text])
    df_no_cause = pd.read_table(os.path.join(sentiment_analysis_data_path, 'emotion_stimulus_no_cause.txt'),
                                delim_whitespace=False, header=0, names=[labeled_text])

    df_dialog = pd.concat([df_cause, df_no_cause])

    df_dialog[ConstantsSentimentAnalysis.TEXT] = df_dialog[labeled_text].apply(lambda x: clean_annotations(x))
    df_dialog[ConstantsSentimentAnalysis.LABEL_EMOTION] = df_dialog[labeled_text].apply(lambda x: get_label(x))
    df_dialog[ConstantsSentimentAnalysis.LABEL_POLARITY] = None
    df_dialog[ConstantsSentimentAnalysis.SOURCE] = ConstantsSentimentDatasetsNames.EMO_STIMULUS
    del df_dialog[labeled_text]
    return df_dialog


def reader_dataset_dialog(sentiment_analysis_data_path):
    """
    Reader for Dialog dataset that include 13117 dialog messages with 102966 annotations (row 671 is a error). The
    dialog messages was split into individual annotations.
    { 0: no emotion, 1: anger, 2: disgust, 3: fear, 4: happiness, 5: sadness, 6: surprise}
    :return:
    """
    emotions_map = {"0": "neutral", "1": "anger", "2": "disgust", "3": "fear", "4": "happiness",
                    "5": "sadness",
                    "6": "surprise"}  # change original label ["no emotion"] to ["Neutral"]
    split_token = '__eou__'
    labeled_text = ConstantsSentimentAnalysis.TEXT
    labeled_emotions = ConstantsSentimentAnalysis.LABEL_EMOTION
    df_dialog_emotion = pd.read_table(os.path.join(sentiment_analysis_data_path, 'dialogues_emotion.txt'),
                                      header=0, names=[labeled_emotions])
    df_dialog_text = pd.read_table(os.path.join(sentiment_analysis_data_path, 'dialogues_text.txt'),
                                   header=0, names=[labeled_text])

    df_dialog_emotion[labeled_emotions] = df_dialog_emotion[labeled_emotions].apply(lambda x: x.split())
    df_dialog_text[labeled_text] = df_dialog_text[labeled_text].apply(
        lambda x: x[0:-len(split_token)].split(split_token))

    index_same_size = df_dialog_emotion[labeled_emotions].apply(lambda x: len(x)) == \
                      df_dialog_text[labeled_text].apply(lambda x: len(x))

    df_dialog_text = df_dialog_text[index_same_size]
    df_dialog_emotion = df_dialog_emotion[index_same_size]

    df_dialog_emotion = df_dialog_emotion.explode(labeled_emotions)
    df_dialog_text = df_dialog_text.explode(labeled_text)

    df_dialog = pd.DataFrame()
    df_dialog[ConstantsSentimentAnalysis.TEXT] = df_dialog_text[ConstantsSentimentAnalysis.TEXT]
    df_dialog[ConstantsSentimentAnalysis.LABEL_EMOTION] = df_dialog_emotion[
        ConstantsSentimentAnalysis.LABEL_EMOTION].replace(emotions_map)
    df_dialog[ConstantsSentimentAnalysis.LABEL_EMOTION] = df_dialog[
        ConstantsSentimentAnalysis.LABEL_EMOTION].apply(lambda x: [x])
    df_dialog[ConstantsSentimentAnalysis.LABEL_POLARITY] = None
    df_dialog[ConstantsSentimentAnalysis.SOURCE] = ConstantsSentimentDatasetsNames.DIALOGUES

    return df_dialog


def text_preprocessing_basic(text: str) -> str:
    """
    Apply basic text process to get only ascii characters, delete repeated white spaces
    and filter punctuation except apostrophe
    :param text:
    :return:
    """

    # BERT transform to unicode  https://github.com/google-research/bert/blob/master/tokenization.py
    # but we are not interested on processing text out of the ascii scope, the ASR only produce ascii
    # By another hand ASR generate text with only one white space, without punctuation (emoticons,", .,/ etc).
    # Lastly the ASR do not generate capital letters for nouns and accents
    # todo parse num to word? Analyze if this has sense and how ASR handle apostrophe ->
    #  check performance with and without contractions -> https://pypi.org/project/pycontractions/

    punctuation_conserve = ["'"]
    table_punctuation = str.maketrans({punctuation: ' ' for punctuation in string.punctuation if punctuation not in
                                       punctuation_conserve})
    text = text.encode("ascii", errors="ignore").decode()
    text = text.translate(table_punctuation)
    # text = re.sub('[\W_]+', ' ', text)  # matches any single letter, number or underscore (same as [a-zA-Z0-9_])
    text = re.sub('\s+', ' ', text)
    text = text.strip()

    return text


def polarity_parser(labels: list) -> str:
    """
    Analyze the list labels to summarize the list content in a string with 3 posibilities
    (1) Use target_polarity_map when all the liste elements are equal
    (2) "empty" when the list elements do not match with any of the target_polarity_map key
    (3) "overlap" when the list elements match with several keys on target_polarity_map
    :param labels:
    :return:
    """
    overlap_label = 'overlap'
    empty_label = 'empty'
    polarities = []
    for label in labels:
        for polarity_category, emotion_list in target_polarity_map.items():
            if label in emotion_list:
                polarities.append(polarity_category)

    unique_polarities = list(set(polarities))

    if len(unique_polarities) == 1:
        return unique_polarities[0]
    elif len(unique_polarities) == 0:
        return empty_label
    else:
        return overlap_label


def filter_letter_sentences(df_sa: pd.DataFrame) -> pd.DataFrame:
    """
    Filter sentence with poor sentences that include few words <No>, many abreviations
    <I got into U.S.C> or many unique characters (regarding the total size) <THEY CALLED ME M I S T E R A S S>
    The method works calculating the ratio between character size and number of words in the sentence
    Rations near to 0.5 means the sentence contain only uni-letter words: 'P E A C E' = 5/9 = 0.55
    0.35 means 35% of the sentence is white space: 'HO NO NO NO NO': 5/15 = 0.35
    s p o o k y , O R B M A I N, S A T I R E, H A M B E R D E R S, S P O I L E R ! !
    :return:
    """
    # todo replace clean_text by text
    threshold = 0.3
    df_sa_ = df_sa.copy()

    df_sa_['token_length'] = df_sa_.text.apply(lambda t: len(t.split()))
    df_sa_['character_length'] = df_sa_.text.apply(lambda x: len(x))
    unwanted_index = (df_sa_.token_length / df_sa_.character_length) > threshold
    df_sa_ = df_sa_[~unwanted_index]

    return df_sa_


def emotions_parser(emotion_list):
    # approach 1. Ekman's map
    """
    # Analysis
    emotions_without_label = []
    for index, row in df_sentiment_analysis.iterrows():
        for emotion in row[ConstantsSentimentAnalysis.LABEL_EMOTION]:
            if emotion in list(chain(*emotions_mask_ekman.values())):
                pass
            else:
                emotions_without_label.append(emotion)
                print(emotion)
    """
    emotion_group_ = []
    emotion_list = set(emotion_list)
    for emotion_group, emotions in emotions_mask_ekman.items():
        if emotions.intersection(emotion_list):
            emotion_group_.append(emotion_group)

    if len(emotion_group_) == 0:
        pass
        #  print(f'label no labeled {emotion_list}')

    return emotion_group_


def get_sentiment_analysis_data(sentiment_analysis_data_path):
    df_isear = reader_dataset_isear(sentiment_analysis_data_path=sentiment_analysis_data_path)
    df_emocap = reader_dataset_emocap(sentiment_analysis_data_path=sentiment_analysis_data_path)
    df_go_emotions = reader_dataset_go_emotions(sentiment_analysis_data_path=sentiment_analysis_data_path)
    df_sa = pd.concat([df_isear, df_emocap, df_go_emotions])
    # Add polarity label
    df_sa[ConstantsSentimentAnalysis.LABEL_POLARITY] = df_sa[ConstantsSentimentAnalysis.LABEL_EMOTION].apply(
        lambda label_emotions: polarity_parser(labels=label_emotions))
    # Compute Ekman emotion for multi label
    df_sa[ConstantsSentimentAnalysis.LABEL_EKMAN] = df_sa[ConstantsSentimentAnalysis.LABEL_EMOTION].apply(
        lambda x: sorted(emotions_parser(x)))  # sorted to keep same cases when data partition
    return df_sa


if __name__ == "__main__":

    DATA_PATH = 'data'
    sentiment_analysis_data_path_ = os.path.join(DATA_PATH, 'raw/sentiment_analysis')

    df_sa_main = get_sentiment_analysis_data(sentiment_analysis_data_path=sentiment_analysis_data_path_)


    df_sentiment_analysis_path_ = os.path.join(DATA_PATH, 'sentiment_analysis.csv')  # to write results
    token_size_values = {'min': 4,
                         'max': 18}  # --> approximated to nearest byte value (to focus on short context-pure)

    df_isear = reader_dataset_isear(sentiment_analysis_data_path=sentiment_analysis_data_path_)
    df_emocap = reader_dataset_emocap(sentiment_analysis_data_path=sentiment_analysis_data_path_)
    df_gne = reader_dataset_good_news_everyone(sentiment_analysis_data_path=sentiment_analysis_data_path_)
    df_go_emotions = reader_dataset_go_emotions(sentiment_analysis_data_path=sentiment_analysis_data_path_)
    df_emo_bank = reader_dataset_emo_bank(sentiment_analysis_data_path=sentiment_analysis_data_path_)
    df_dialog = reader_dataset_dialog(sentiment_analysis_data_path=sentiment_analysis_data_path_)
    df_emotion_stimulus = reader_dataset_emotion_stimulus(sentiment_analysis_data_path=sentiment_analysis_data_path_)
    df_sentiment_analysis = pd.concat([df_isear, df_emocap, df_gne, df_go_emotions, df_emo_bank,
                                       df_emotion_stimulus, df_dialog
                                       ])
    df_sentiment_analysis[ConstantsSentimentAnalysis.TEXT] = df_sentiment_analysis.text.apply(
        lambda text: text_preprocessing_basic(text))
    df_sentiment_analysis = filter_letter_sentences(df_sa=df_sentiment_analysis)  # --> v1:745 v2:2794 delete

    df_sentiment_analysis = df_sentiment_analysis[(df_sentiment_analysis['token_length'] >= token_size_values['min'])
                                                  & (df_sentiment_analysis['token_length'] <= token_size_values['max'])
                                                  | ((df_sentiment_analysis.source.isin(
        [ConstantsSentimentDatasetsNames.ISEAR])  # very useful dataset
                                                     ) & (df_sentiment_analysis['token_length'] >= 3)
                                                     )
                                                  ]

    df_sentiment_analysis[ConstantsSentimentAnalysis.LABEL_POLARITY] = df_sentiment_analysis[
        ConstantsSentimentAnalysis.LABEL_EMOTION].apply(lambda label_emotions: polarity_parser(labels=label_emotions))

    cases = [['excitement', 'neutral'], ['anger', 'love'], ['admiration', 'disappointment'],
             ['nervousness', 'neutral'], ['love', 'sadness'], ['disapproval', 'love'], ['anger', 'neutral'],
             ['fear', 'neutral']]

    df_sentiment_analysis.to_csv(df_sentiment_analysis_path_, sep=';', index=False)

