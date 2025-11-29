from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_tokenizer(files, save_path="tokenizer.json", vocab_size=30000):
    print("--- Training BPE Tokenizer ---")
    
    # 1. Initialize an empty BPE Tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # 2. Define Special Tokens
    # [PAD]: Padding (0)
    # [SOS]: Start of Sequence (1)
    # [EOS]: End of Sequence (2)
    # [UNK]: Unknown word (3)
    special_tokens = ["[PAD]", "[SOS]", "[EOS]", "[UNK]"]
    
    # 3. Configure the Trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=special_tokens,
        min_frequency=2 # Ignore words that appear only once
    )
    
    # 4. Train on both files (Source and Target)
    # We combine them so the model understands English generally
    tokenizer.train(files, trainer)
    
    # 5. Save
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path} with vocab size {tokenizer.get_vocab_size()}")
    
    return tokenizer

if __name__ == "__main__":
    files = ["src_train.txt", "trg_train.txt"]
    
    # Train
    tokenizer = train_tokenizer(files)
    
    # Test it immediately
    sentence = "Gilded flowers, tapestry pillows, fresh oranges in a glass bowl, and a bottle of spirits suggest Tuscan decor and home accents."
    encoded = tokenizer.encode(sentence)
    
    print("\n--- Validation ---")
    print(f"Input: '{sentence}'")
    print(f"Tokens: {encoded.tokens}")
    print(f"IDs:    {encoded.ids}")
    print(F"Number of tokens: {len(encoded.ids)}")