import config
from transformers import BertForTokenClassification
import torch.nn as nn

class NERModel(nn.Module):
    def __init__(self, num_labels):
        super(NERModel, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=num_labels)

    def forward(self, input_ids, mask, labels=None):
        return self.bert(input_ids=input_ids, attention_mask=mask, labels=labels, return_dict=False)











