import torch
from transformers import AutoModel


class RobertaMultiLabel(torch.nn.Module):
    def __init__(self, config: ConfigSeq2SeqMultiLabel):
        super(RobertaMultiLabel, self).__init__()
        self.model_name = config.MODEL_NAME
        self.model = AutoModel.from_pretrained(self.model_name, return_dict=False)
        self.dropout_layer = torch.nn.Dropout(config.MODEL_DROPOUT)
        self.linear_layer = torch.nn.Linear(config.MODEL_HIDDEN_STATES, config.MODEL_LABELS)

    def forward(self, input_ids, attention_mask):
        last_hidden_states, pooling_layer = self.model(input_ids=input_ids, attention_mask=attention_mask)
        output = self.dropout_layer(pooling_layer)
        output = self.linear_layer(output)
        return output