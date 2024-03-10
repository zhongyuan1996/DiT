from transformers import BertModel, BertConfig
import torch.nn as nn
import torch

class splitBERT(nn.Module):
    def __init__(self, bert, config, split_at):
        super(splitBERT, self).__init__()
        self.bert = bert
        self.config = config
        self.split_at = split_at

        self.encoder_layers = nn.ModuleList(self.bert.encoder.layer[:split_at])
        self.decoder_layers = nn.ModuleList(self.bert.encoder.layer[split_at:])

        self.prediction_head = nn.Linear(self.config.hidden_size, self.config.vocab_size)
        self.next_visit_head = nn.Linear(self.config.hidden_size, self.config.vocab_size)

    def encoder_forward(self, input_ids, attention_mask = None):
        embeddings = self.bert.embeddings(input_ids)
        encoder_output = embeddings
        for layer_module in self.encoder_layers:
            layer_outputs = layer_module(encoder_output, attention_mask)
            encoder_output = layer_outputs[0]
        return encoder_output

    def decoder_forward(self, encoder_output, attention_mask = None):
        decoder_output = encoder_output
        for layer_module in self.decoder_layers:
            layer_outputs = layer_module(decoder_output, attention_mask)
            decoder_output = layer_outputs[0]
        return decoder_output

    def direct_forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):

        encoder_output = self.encoder_forward(input_ids, attention_mask)
        decoder_output = self.decoder_forward(encoder_output, attention_mask)

        prediction_scores = self.prediction_head(decoder_output)
        next_visit_scores = self.next_visit_head(decoder_output)

        return prediction_scores, next_visit_scores
