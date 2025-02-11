# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
###---------------------Packing------------------------------------------
from tqdm import tqdm
from itertools import chain

from torch.utils.data import Dataset

class PackConcatDataset(Dataset):
    def __init__(self, dataset, chunk_size=4096, tokenizer=None):
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.tokenizer = tokenizer
        self.samples = []
        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        
        for i, sample in enumerate(tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True)):
            buffer = {k: v + sample[k] for k, v in buffer.items()}
            
            while len(next(iter(buffer.values()))) > self.chunk_size:
                # Store the chunk
                chunk = {k: v[:self.chunk_size] for k, v in buffer.items()}
                self.samples.append(chunk)
                
                # Print two consecutive chunks for debugging
                if len(self.samples) > 1 and len(self.samples) <= 3 and self.tokenizer:
                    print(f"\nChunk {len(self.samples)-1}:")
                    print(self.tokenizer.decode(chunk['input_ids']))
                    print("\nTransition to next chunk:")
                    next_text = self.tokenizer.decode(buffer['input_ids'][self.chunk_size:self.chunk_size+100])
                    print(next_text[:200] + "...")
                    
                # Update buffer
                buffer = {k: v[self.chunk_size:] for k, v in buffer.items()}

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


##------------- padding----------------

from tqdm import tqdm
from torch.utils.data import Dataset

class PadConcatDataset(Dataset):
    def __init__(self, dataset, chunk_size=4096, tokenizer=None):
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.tokenizer = tokenizer
        self.samples = []
        
        for i, sample in enumerate(tqdm(self.dataset, desc="Processing dataset", dynamic_ncols=True)):
            # Get the current length of the sequence
            seq_length = len(sample['input_ids'])
            
            if seq_length < chunk_size:
                # Calculate padding length
                pad_length = chunk_size - seq_length
                
                # Pad input_ids with tokenizer pad token or 0
                pad_token = self.tokenizer.pad_token_id if tokenizer else 0
                padded_input_ids = sample['input_ids'] + [pad_token] * pad_length
                
                # Pad attention mask with 0s
                padded_attention_mask = sample['attention_mask'] + [0] * pad_length
                
                # Pad labels with -100 (typical ignore index for CrossEntropyLoss)
                padded_labels = sample['labels'] + [-100] * pad_length
                
                padded_sample = {
                    'input_ids': padded_input_ids,
                    'attention_mask': padded_attention_mask,
                    'labels': padded_labels
                }
                self.samples.append(padded_sample)
                
                # Print debug info for first few samples
                if i < 2 and self.tokenizer:
                    print(f"\nSample {i+1}:")
                    print(f"Original length: {seq_length}")
                    print(f"Padded to: {chunk_size}")
                    print("Text content:")
                    print(self.tokenizer.decode(sample['input_ids']))
                    print(f"Padding added: {pad_length} tokens")
            
            elif seq_length == chunk_size:
                # If already at chunk_size, add as is
                self.samples.append(sample)
            
            else:
                # If longer than chunk_size, warn and truncate
                print(f"Warning: Sample {i} exceeds chunk_size ({seq_length} > {chunk_size}). Truncating.")
                truncated_sample = {
                    'input_ids': sample['input_ids'][:chunk_size],
                    'attention_mask': sample['attention_mask'][:chunk_size],
                    'labels': sample['labels'][:chunk_size]
                }
                self.samples.append(truncated_sample)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)



### function calling
# context length is the supported lenth: for llama3.1 its 128k, however, this length depends on available GPU RAM
# if GPU is low RAM, then choose low context length to fit the GPU
dataset_train = PackConcatDataset(
                dataset_train, chunk_size=train_config.context_length,
                tokenizer=tokenizer

dataset_train = PadConcatDataset(
                dataset_train, chunk_size=train_config.context_length,
                tokenizer=tokenizer
