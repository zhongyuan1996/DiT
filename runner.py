# Assuming you're using a BertTokenizer
from transformers import BertTokenizer, BertModel, BertConfig
import pandas as pd
from dataloader import ehrDataset
from torch.utils.data import Dataset, DataLoader
import pickle
from mainModel import splitBERT


# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./pretrained_models')

# Load extra tokens and add them to the tokenizer
extra_tokens = pickle.load(open('./data/ehr/mimic3_vocab.pkl', 'rb'))
tokenizer.add_tokens(list(extra_tokens))

# Define and add special tokens
special_tokens_dict = {
    'additional_special_tokens': ['[DIAG]', '[PROC]', '[DRG]', '[LAB]', '[EOP]']
}
tokenizer.add_special_tokens(special_tokens_dict)

# Initialize the BERT model
bert = BertModel.from_pretrained('bert-base-uncased', cache_dir='./pretrained_models')
bert.resize_token_embeddings(len(tokenizer))  # Resize embeddings to match the tokenizer

# Update the configuration
config = BertConfig.from_pretrained('bert-base-uncased', cache_dir='./pretrained_models')
config.vocab_size = len(tokenizer)  # Ensure config reflects the new vocab size

# Initialize your custom splitBERT model
split_Bert = splitBERT(bert, config, 6)


df = pd.read_csv('./data/ehr/mimic3_ehr_toy.csv')
dataset = ehrDataset(df, tokenizer)
loader = DataLoader(dataset, batch_size=5, shuffle=True)

for batch in loader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    print("Batch input_ids shape:", input_ids.shape)  # Example processing
    print("Batch labels:", labels)