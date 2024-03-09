from transformers import BertModel, BertConfig
import torch.nn as nn
import torch

class splitBERT(nn.Module):
    def __init__(self, model_name_path, split_at):
        super(splitBERT, self).__init__()
        self.bert = BertModel.from_pretrained(model_name_path)
        self.config = BertConfig.from_pretrained(model_name_path)
        self.split_at = split_at

        self.encoder_layers = nn.ModuleList(self.bert.encoder.layer[:split_at])
        self.decoder_layers = nn.ModuleList(self.bert.encoder.layer[split_at:])

