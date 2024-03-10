import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from random import random
from ast import literal_eval


class ehrDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512, mask_probability=0.15, mask_next_visit_probability=0.8):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_probability = mask_probability
        self.mask_next_visit_probability = mask_next_visit_probability
        self.dataset = []
        self.special_token_ids = {self.tokenizer.cls_token_id, self.tokenizer.sep_token_id,
                             self.tokenizer.convert_tokens_to_ids('[EOP]')}

        for _, row in dataframe.iterrows():
            diagnoses = literal_eval(row['DIAGNOSES'])
            proc_item = literal_eval(row['PROC_ITEM'])
            drg_code = literal_eval(row['DRG_CODE'])
            lab_item = literal_eval(row['LAB_ITEM'])
            mortality = row['MORTALITY']

            num_visits = len(diagnoses)
            for visit_idx in range(num_visits):
                current_visit_text = self.compose_visit_text(diagnoses, proc_item, drg_code, lab_item, visit_idx)
                next_visit_text = self.compose_visit_text(diagnoses, proc_item, drg_code, lab_item,
                                                          visit_idx + 1) if visit_idx < num_visits - 1 else "[EOP]"

                input_ids, labels_mlm = self.prepare_visits(current_visit_text, next_visit_text,
                                                                               visit_idx, num_visits)

                # Create attention mask
                attention_mask = [1] * len(input_ids) + [0] * (self.max_length - len(input_ids))
                input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
                labels_mlm += [-100] * (self.max_length - len(labels_mlm))

                self.dataset.append({
                    'input_ids': torch.tensor(input_ids, dtype=torch.long),
                    'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                    'labels_mlm': torch.tensor(labels_mlm, dtype=torch.long),
                    'labels_mortality': torch.tensor(mortality, dtype=torch.long),
                })

    def compose_visit_text(self, diagnoses, proc_item, drg_code, lab_item, visit_idx):
        if visit_idx >= len(diagnoses): return "[EOP]"
        return '[DIAG] ' + ' '.join(diagnoses[visit_idx]) + \
            ' [PROC] ' + ' '.join(proc_item[visit_idx]) + \
            ' [DRG] ' + ' '.join(drg_code[visit_idx]) + \
            ' [LAB] ' + ' '.join(lab_item[visit_idx])

    def prepare_visits(self, current_visit, next_visit, visit_idx, num_visits):
        # Tokenize current and next visit
        current_visit_tokens = self.tokenizer.tokenize(current_visit)
        next_visit_tokens = self.tokenizer.tokenize(next_visit) if next_visit != "[EOP]" else self.tokenizer.tokenize("[EOP]")

        # Mask tokens in the current visit for MLM
        input_ids, labels_mlm = self.mask_tokens(current_visit_tokens)

        # Decide whether to mask the entire next visit
        mask_next_visit = random() < self.mask_next_visit_probability and visit_idx < num_visits - 1

        if mask_next_visit:
            # Prepare masked input IDs for the entire next visit
            next_visit_input_ids = [self.tokenizer.mask_token_id] * len(next_visit_tokens)
            # Update labels for MLM with the true labels of the masked next visit tokens
            labels_for_next_visit = self.tokenizer.convert_tokens_to_ids(next_visit_tokens)
        else:
            next_visit_input_ids = self.tokenizer.convert_tokens_to_ids(next_visit_tokens)
            # If the next visit is not masked, no need to predict its tokens, so set labels to -100
            labels_for_next_visit = [-100] * len(next_visit_tokens)

        # Combine current and next visit inputs
        combined_input_ids = input_ids + [self.tokenizer.sep_token_id] + next_visit_input_ids
        # Adjust labels for MLM accordingly, including the SEP token between visits
        labels_mlm = labels_mlm + [-100] + labels_for_next_visit

        return combined_input_ids, labels_mlm

    def mask_tokens(self, tokenized_text):
        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        labels = [-100] * len(input_ids)  # Initialize labels with -100

        for i in range(len(input_ids)):
            # Skip masking if the token is a special token
            if input_ids[i] in self.special_token_ids:
                continue

            # Mask token with a given probability
            if random() < self.mask_probability:
                labels[i] = input_ids[i]  # Original token is used as a label
                input_ids[i] = self.tokenizer.mask_token_id  # Replace with mask token ID

        return input_ids, labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
