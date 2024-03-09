from torch.utils.data import Dataset
import torch
import pandas as pd
from ast import literal_eval
class ehrDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items = []

        for _, row in dataframe.iterrows():
            diagnoses = literal_eval(row['DIAGNOSES'])
            proc_item = literal_eval(row['PROC_ITEM'])
            drg_code = literal_eval(row['DRG_CODE'])
            lab_item = literal_eval(row['LAB_ITEM'])
            mortality = row['MORTALITY']

            for visit_idx in range(len(diagnoses)):
                # Concatenate visit data with special tokens
                visit_text = '[DIAG] ' + ' '.join(diagnoses[visit_idx]) + \
                             ' [PROC] ' + ' '.join(proc_item[visit_idx]) + \
                             ' [DRG] ' + ' '.join(drg_code[visit_idx]) + \
                             ' [LAB] ' + ' '.join(lab_item[visit_idx])

                # Tokenize the visit text
                inputs = self.tokenizer.encode_plus(
                    visit_text,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                )

                self.items.append({
                    'input_ids': inputs['input_ids'].squeeze(0),  # Remove batch dimension
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(mortality, dtype=torch.long)
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


