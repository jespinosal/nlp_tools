import torch
import pandas as pd
from transformers import AutoTokenizer
from config import ConfigSeq2SeqMultiLabel


class DataSetSeq2SeqMultiLabel(torch.utils.data.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 tokenizer: AutoTokenizer,
                 config: ConfigSeq2SeqMultiLabel,
                 inference_mode: bool = False,
                 text_column: str = ''
                 ):
        self.df = df
        self.inference_mode = inference_mode
        self.tokenizer = tokenizer
        self.max_length = config.MAX_LENGTH
        self.padding_strategy = 'max_length'
        self.config = config
        self.text_column = text_column

    def __getitem__(self, idx):
        text = self.df[self.text_column].iloc[idx]
        labels_encoded = torch.Tensor([float('NaN')]) if self.inference_mode else \
        self.df[self.config.LABEL_COLUMNS].iloc[idx].values
        input_encoded = self.tokenizer.encode_plus(text,
                                                   truncation=True,
                                                   add_special_tokens=True,
                                                   max_length=self.max_length,
                                                   padding=self.padding_strategy)

        return {'input_ids': torch.tensor(input_encoded['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(input_encoded['attention_mask'], dtype=torch.long),
                'labels': torch.Tensor([float('NaN')]) if self.inference_mode else torch.tensor(labels_encoded,
                                                                                                dtype=torch.float)}

    def __len__(self):
        return len(self.df)
