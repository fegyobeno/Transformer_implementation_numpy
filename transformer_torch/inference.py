import torch
import torch.nn as nn
from tokenizers import Tokenizer
from transformer import Transformer # Ensure this imports your class

# --- 1. Configuration (MUST MATCH TRAINING) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "TRAIN_1/transformer_epoch_91.pth" # Or whichever epoch you want to test
TOKENIZER_PATH = "tokenizer.json"

# Architecture Specs (Update these if you used the 2060 Optimized configs)
D_MODEL = 512  # Reduced from 512 for faster training on smaller data
HEADS = 8
N_LAYERS = 6   # Reduced from 6
D_FF = 1024
DROPOUT = 0.1

def predict(model, sentence, tokenizer, device, max_length=50):
    model.eval()
    
    # Encode
    ids = tokenizer.encode(sentence).ids
    ids = [tokenizer.token_to_id("[SOS]")] + ids + [tokenizer.token_to_id("[EOS]")]
    src_tensor = torch.tensor(ids).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_out = model.encoder(src_tensor, src_mask)
    
    # Decode
    trg_indices = [tokenizer.token_to_id("[SOS]")]
    
    for i in range(max_length):
        trg_tensor = torch.tensor(trg_indices).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output = model.decoder(trg_tensor, enc_out, src_mask, trg_mask)
            logits = model.out_linear(output[:, -1, :])
            next_token_id = logits.argmax(dim=-1).item()
            
        if next_token_id == tokenizer.token_to_id("[EOS]"):
            break
            
        trg_indices.append(next_token_id)
        
    return tokenizer.decode(trg_indices[1:], skip_special_tokens=True)

import torch.nn.functional as F
import math

def beam_search_decode(model, sentence, tokenizer, device, beam_width=3, max_len=50):
    """
    Performs Beam Search to generate better translations/paraphrases.
    """
    model.eval()
    
    # 1. Encode Source (Do this only once!)
    # -------------------------------------
    ids = tokenizer.encode(sentence).ids
    ids = [tokenizer.token_to_id("[SOS]")] + ids + [tokenizer.token_to_id("[EOS]")]
    src_tensor = torch.tensor(ids).unsqueeze(0).to(device) # (1, Seq_Len)
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_out = model.encoder(src_tensor, src_mask)
        
    # 2. Initialize Beams
    # -------------------------------------
    # Each beam is a tuple: (sequence_tensor, score)
    # Start with just the [SOS] token and a score of 0.0
    sos_token = tokenizer.token_to_id("[SOS]")
    eos_token = tokenizer.token_to_id("[EOS]")
    
    # List of active candidates
    beams = [(torch.tensor([sos_token], device=device), 0.0)]
    
    # List of completed sequences
    completed = []
    
    # 3. The Search Loop
    # -------------------------------------
    for _ in range(max_len):
        candidates = []
        
        # Expand each current beam
        for seq, score in beams:
            # If this beam is already done (hit EOS), keep it but don't expand
            # (Logic handled by moving to 'completed' list below)
            
            # Prepare input for decoder
            trg_tensor = seq.unsqueeze(0) # Add batch dim: (1, Len)
            trg_mask = model.make_trg_mask(trg_tensor)
            
            with torch.no_grad():
                # Forward pass
                # We reuse enc_out (Cross-Attention)
                out = model.decoder(trg_tensor, enc_out, src_mask, trg_mask)
                
                # Get logits for the LAST token only
                logits = model.out_linear(out[:, -1, :])
                
                # Convert to Log-Probabilities (Crucial for Beam Search)
                # shape: (1, Vocab_Size)
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Get top k probable next tokens to save time
                # We take 'beam_width' best options
                top_scores, top_indices = torch.topk(log_probs, beam_width)
                
            # Create new candidates
            for i in range(beam_width):
                token_id = top_indices[0][i].item()
                token_score = top_scores[0][i].item()
                
                new_seq = torch.cat([seq, torch.tensor([token_id], device=device)])
                new_score = score + token_score # Add log-prob
                
                # Check for EOS
                if token_id == eos_token:
                    # Normalize score by length so short sentences aren't unfairly favored
                    # (Simple length penalty: score / length)
                    final_score = new_score / len(new_seq)
                    completed.append((new_seq, final_score))
                else:
                    candidates.append((new_seq, new_score))
        
        # 4. Pruning (Natural Selection)
        # -------------------------------------
        # Sort candidates by score (highest is best, since log_probs are negative close to 0)
        # We want the values closest to 0 (e.g., -1.5 is better than -50.0)
        ordered = sorted(candidates, key=lambda x: x[1], reverse=True)
        
        # Keep only the top 'beam_width' alive
        beams = ordered[:beam_width]
        
        # Stop early if we have enough completed sentences
        if len(completed) >= beam_width:
            break
            
    # 5. Final Selection
    # -------------------------------------
    # If no sequence finished naturally, treat current beams as finished
    if len(completed) == 0:
        completed = beams
        
    # Sort completed by normalized score
    # Note: If we use the fallback 'beams', we might want to normalize their scores too
    best_seq, best_score = max(completed, key=lambda x: x[1])
    
    # Decode to text
    return tokenizer.decode(best_seq.tolist(), skip_special_tokens=True)

def main():
    print("--- Loading System ---")
    
    # 1. Load Tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    pad_idx = tokenizer.token_to_id("[PAD]")
    
    print(f"Vocab Size: {vocab_size}")
    
    # 2. Rebuild Architecture
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
    # Intermittent step, summarise the model params
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model built with {total_params} trainable parameters.")
    # 3. Load Weights
    print(f"Loading weights from {CHECKPOINT_PATH}...")
    try:
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Did you change hyperparams? The model architecture must match the saved file exactly.")
        return

    # 4. Interactive Loop
    print("\n--- Model Ready (Type 'q' to quit) ---")
    while True:
        text = input("Enter sentence to paraphrase: ")
        if text.lower() == 'q':
            break
            
        #response = predict(model, text, tokenizer, DEVICE)
        response = beam_search_decode(model, text, tokenizer, DEVICE, beam_width=5)
        print(f" > {response}\n")

if __name__ == "__main__":
    main()