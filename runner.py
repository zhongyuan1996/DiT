# Assuming you're using a BertTokenizer
from transformers import BertTokenizer, BertModel
import pandas as pd
from dataloader import ehrDataset
from torch.utils.data import Dataset, DataLoader
import pickle

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',cache_dir = './pretrained_models')

# Define special tokens for each modality
special_tokens_dict = {
    'additional_special_tokens': ['[DIAG]', '[PROC]', '[DRG]', '[LAB]']
}

#extra tokens in file
extra_tokens = pickle.load(open('./data/ehr/mimic3_vocab.pkl','rb'))
tokenizer.add_tokens(list(extra_tokens))

num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

model = BertModel.from_pretrained('bert-base-uncased')
model.resize_token_embeddings(len(tokenizer))

df = pd.read_csv('./data/ehr/mimic3_ehr_toy.csv')
dataset = ehrDataset(df, tokenizer)
loader = DataLoader(dataset, batch_size=5, shuffle=True)

for batch in loader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    print("Batch input_ids shape:", input_ids.shape)  # Example processing
    print("Batch labels:", labels)