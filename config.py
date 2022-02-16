from typing import List


class ConfigSeq2SeqMultiLabel:
    DEVICE: List[str] = 'cpu'
    #  data
    LABEL_COLUMNS: list = ['']
    TEXT_COLUMN: str = 'text'
    PARITION_COLUMN = ''
    # model
    MODEL_NAME: str = '../input/roberta-base'
    MODEL_DROPOUT: float = 0.2
    MODEL_HIDDEN_STATES: int = 768
    MODEL_LABELS: int = len(LABEL_COLUMNS)
    # tokenizer
    MAX_LENGTH: int = 128
    # train loop config
    TRAIN_BATCH_SIZE: int = 64
    LEARNING_RATE: float = 3e-5
    EVAL_BATCH_SIZE: int = 64
    TEST_BATCH_SIZE: int = 64
    EPOCHS: int = 3
