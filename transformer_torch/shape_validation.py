from transformer import *
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

def main():
    print("--- Testing Pipeline: Input -> PE -> Attention ---")
    
    # 1. Setup Dimensions
    batch_size = 2
    seq_len = 10
    d_model = 64  # This acts as our embedding size
    heads = 8
    d_ff = 256
    dropout = 0.0
    N = 2  # Number of Encoder/Decoder layers

    # 2. Input: A raw sequence of embeddings
    # Shape: (Batch, Seq, d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"1. Original Input Shape:    {x.shape}")

    # 3. Apply Positional Encoding
    # Logic: Inject order into the embeddings
    pos_encoder = PositionalEncoding(d_model=d_model, max_len=50)
    x_pe = pos_encoder(x)
    print(f"2. After Positional Enc:    {x_pe.shape}")
    print(f"Pos encoding values (first batch, first position): {x_pe[0,0,:5]}")

    # 4. Prepare for Scaled Dot-Product Attention
    # The Attention module expects (Batch, Heads, Seq, Dim).
    # Since we haven't built the Projection layers (MultiHead) yet,
    # we manually reshape 'x_pe' to simulate 1 Head.
    # New Shape: (Batch, 1, Seq, d_model)
    q = k = v = x_pe.unsqueeze(1) 
    print(f"3. Reshaped for Attention:  {q.shape} (Simulating 1 Head)")

    # 5. Apply Attention (Self-Attention)
    # Logic: The sequence looks at itself to find context
    attention = ScaledDotProductAttention(dropout=dropout)
    output, weights = attention(q, k, v)

    # 6. Squeeze back to original rank for viewing
    output = output.squeeze(1)

    print(f"4. Final Output Shape:      {output.shape}")
    print(f"5. Attention Weights:       {weights.shape}")
    
    
    multi_head_attention = MultiHeadAttention(d_model=d_model, h=heads, dropout=dropout)
    output_mha, weights_mha = multi_head_attention(x_pe, x_pe, x_pe)
    print(f"Input Shape:  {x_pe.shape}")
    print(f"Output Shape: {output_mha.shape}")
    
    FF = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
    output_ff = FF(output_mha)
    print(f"FeedForward Output Shape: {output_ff.shape}")
    
    encoder_layer = EncoderLayer(d_model=d_model, heads=heads, d_ff=d_ff, dropout=dropout)
    output_enc = encoder_layer(x_pe, mask=None)
    print(f"Encoder Layer Output Shape: {output_enc.shape}")
    
    y = torch.randn(batch_size, seq_len, d_model)
    decoder_layer = DecoderLayer(d_model=d_model, heads=heads, d_ff=d_ff, dropout=dropout)
    output_dec = decoder_layer(y, output_enc, src_mask=None, tgt_mask=None)
    print(f"Decoder Layer Output Shape: {output_dec.shape}")
    
    encoder = Encoder(vocab_size=1000, d_model=d_model, N=N, heads=heads, d_ff=d_ff, dropout=dropout)
    inp = torch.randint(0, 1000, (batch_size, seq_len))
    output_enc_full = encoder(inp, mask=None)
    print(f"Full Encoder Output Shape: {output_enc_full.shape}")
    decoder = Decoder(vocab_size=1000, d_model=d_model, N=N, heads=heads, d_ff=d_ff, dropout=dropout)
    output_dec_full = decoder(inp, output_enc_full, src_mask=None, tgt_mask=None)
    print(f"Full Decoder Output Shape: {output_dec_full.shape}")
    
    # Validation
    assert output.shape == (batch_size, seq_len, d_model)
    assert weights.shape == (batch_size, 1, seq_len, seq_len)
    assert output_mha.shape == (batch_size, seq_len, d_model)
    assert output_ff.shape == (batch_size, seq_len, d_model)
    assert output_enc.shape == (batch_size, seq_len, d_model)
    assert output_dec.shape == (batch_size, seq_len, d_model)
    assert output_enc_full.shape == (batch_size, seq_len, d_model)
    assert output_dec_full.shape == (batch_size, seq_len, d_model)
    print("\nTest Passed: Data flowed successfully from Embedding to Attention.")
    
if __name__ == "__main__":
    main()