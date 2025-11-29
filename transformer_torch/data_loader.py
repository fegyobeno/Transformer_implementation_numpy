import torch
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

class ParaphraseDataset(Dataset):
    def __init__(self, src_file, trg_file, tokenizer_path="tokenizer.json", max_len=64):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len = max_len
        
        # Load Raw Text
        with open(src_file, 'r', encoding='utf-8') as f:
            self.src_lines = f.read().splitlines()
        with open(trg_file, 'r', encoding='utf-8') as f:
            self.trg_lines = f.read().splitlines()
            
        assert len(self.src_lines) == len(self.trg_lines), "Source and Target files must have same length!"
        
        # Pre-fetch IDs for special tokens
        self.sos_token = self.tokenizer.token_to_id("[SOS]")
        self.eos_token = self.tokenizer.token_to_id("[EOS]")
        self.pad_token = self.tokenizer.token_to_id("[PAD]")

    def __len__(self):
        return len(self.src_lines)
    
    def __getitem__(self, idx):
        src_text = self.src_lines[idx]
        trg_text = self.trg_lines[idx]
        
        # 1. Tokenize (Text -> List of IDs)
        # enable_padding=False here because we will pad dynamically in the DataLoader (Collate)
        src_encoded = self.tokenizer.encode(src_text).ids
        trg_encoded = self.tokenizer.encode(trg_text).ids
        
        # 2. Add SOS and EOS
        # Source usually doesn't need SOS for Encoder, but it doesn't hurt.
        # Target DEFINITELY needs SOS (start signal) and EOS (stop signal).
        
        # Truncate if too long (leave space for special tokens)
        if len(src_encoded) > self.max_len - 2: src_encoded = src_encoded[:self.max_len - 2]
        if len(trg_encoded) > self.max_len - 2: trg_encoded = trg_encoded[:self.max_len - 2]
        
        # Format: [SOS] + Tokens + [EOS]
        src_ids = [self.sos_token] + src_encoded + [self.eos_token]
        trg_ids = [self.sos_token] + trg_encoded + [self.eos_token]
        
        return torch.tensor(src_ids), torch.tensor(trg_ids)

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    
    # Pad sequences in this specific batch to the length of the longest one in this batch
    # padding_value=0 assumes [PAD] is ID 0
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    trg_padded = torch.nn.utils.rnn.pad_sequence(trg_batch, batch_first=True, padding_value=0)
    
    return src_padded, trg_padded
