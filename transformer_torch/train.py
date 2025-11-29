import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm # For progress bars
from transformer import Transformer
from data_loader import ParaphraseDataset, collate_fn

def train_model():
    # 1. Configuration / Hyperparameters
    # ----------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 512
    EPOCHS = 4
    LEARNING_RATE = 0.0001
    MAX_LEN = 64
    
    # Model Specs (Small/Medium for COCO)
    D_MODEL = 512  # Reduced from 512 for faster training on smaller data
    HEADS = 8
    N_LAYERS = 6   # Reduced from 6
    D_FF = 1024
    DROPOUT = 0.1

    print(f"--- Training Configuration ---")
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Model Dim: {D_MODEL}")
    
    # 2. Preparation
    # ----------------------------------
    # Load Tokenizer to get vocab size and special IDs
    tokenizer = Tokenizer.from_file("tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()
    pad_idx = tokenizer.token_to_id("[PAD]")
    
    # Create Dataset & Loader
    dataset = ParaphraseDataset(
        src_file="src_train.txt", 
        trg_file="trg_train.txt", 
        tokenizer_path="tokenizer.json",
        max_len=MAX_LEN
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2,   # Use 0 if on Windows and getting errors
        pin_memory=True
    )
    
    # Instantiate Model
    model = Transformer(
        src_vocab_size=vocab_size,
        trg_vocab_size=vocab_size,
        src_pad_idx=pad_idx,
        trg_pad_idx=pad_idx,
        d_model=D_MODEL,
        N=N_LAYERS,
        heads=HEADS,
        d_ff=D_FF,
        dropout=DROPOUT,
        device=DEVICE
    ).to(DEVICE)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    
    # Loss Function
    # We ignore the [PAD] index so the model doesn't cheat by predicting padding
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    # 3. Training Loop
    # ----------------------------------
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for src, trg in pbar:
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            
            # --- Input Shifting (Teacher Forcing) ---
            # Trg Input:  [SOS, A, B, C] (Feed this to decoder)
            # Trg Target: [A, B, C, EOS] (Predict this)
            
            trg_input = trg[:, :-1] # Remove last token (EOS or Pad)
            trg_target = trg[:, 1:] # Remove first token (SOS)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward Pass
            # The model predicts the next token for every position in trg_input
            output = model(src, trg_input)
            
            # --- Reshape for Loss ---
            # Output: (Batch, Seq_Len, Vocab) -> (Batch * Seq_Len, Vocab)
            # Target: (Batch, Seq_Len)        -> (Batch * Seq_Len)
            output_reshaped = output.contiguous().view(-1, vocab_size)
            trg_target_reshaped = trg_target.contiguous().view(-1)
            
            # Calculate Loss
            loss = criterion(output_reshaped, trg_target_reshaped)
            
            # Backward Pass
            loss.backward()
            
            # Gradient Clipping (Prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update Progress Bar
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"End of Epoch {epoch+1} | Average Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        torch.save(model.state_dict(), f"transformer_epoch_{epoch+1}.pth")

    print("Training Complete!")
    
def main():
    train_model()

if __name__ == "__main__":
    main()