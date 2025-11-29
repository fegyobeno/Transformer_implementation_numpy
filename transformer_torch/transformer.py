import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# set device to cuda:0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant 'pe' matrix with values dependent on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate the division term: 10000^(2i/d_model)
        # Implemented in log space for numerical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as a buffer (part of state_dict, but not a learnable parameter)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        """
        Input: (batch_size, seq_len, d_model)
        Adds positional encoding to the input embeddings.
        """
        # Slice pe to the current sequence length
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Scaled Dot-Product Attention Module
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, q : torch.Tensor, k : torch.Tensor, v : torch.Tensor, mask : torch.Tensor = None) -> torch.Tensor:
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)
        
        attention = self.softmax(scores)
        attention = self.dropout(attention)
        output = torch.matmul(attention, v)
        
        return output, attention
    
# Multi-Head Attention Module
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float = 0.0, mask: torch.Tensor = None):
        super().__init__()
        assert d_model % h == 0, "The model dimension must be divisible by the number of heads (h)"
        
        self.d_k = d_model // h
        self.h = h
        # Linear projections for Query, Key, Value
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        # Final output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout=dropout)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        """
        Input Shape: (Batch, Seq_Len, d_model)
        Output Shape: (Batch, Seq_Len, d_model)
        """
        batch_size = q.size(0)
        
        # 1. Linear Projection & Split Heads
        # Reshape: (Batch, Seq, h, d_k) -> Transpose: (Batch, h, Seq, d_k)
        # We allow different sequence lengths for q and k/v (cross-attention)
        q = self.w_q(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        
        # 2. Scaled Dot-Product Attention
        # Output: (Batch, h, Seq, d_k)
        x, attn_weights = self.attention(q, k, v, mask=mask)
        
        # 3. Concatenate Heads
        # Transpose: (Batch, Seq, h, d_k) -> Contiguous -> View: (Batch, Seq, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        
        # 4. Final Linear Projection
        return self.w_o(x), attn_weights
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.relu(self.linear1(x))))
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model : int, heads : int, d_ff : int, dropout : float = 0.0):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 1. Self-Attention Sublayer
        # Residual connection happens *after* dropout, then Norm
        # paper: LayerNorm(x + Sublayer(x))
        attn_out, _ = self.mha(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 2. Feed Forward Sublayer
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model : int, heads : int, d_ff : int, dropout : float = 0.0):
        super(DecoderLayer, self).__init__()
        # 1. Masked Self-Attention (Target attends to Target)
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        
        # 2. Cross-Attention (Target attends to Source)
        self.cross_attn = MultiHeadAttention(d_model, heads, dropout)
        
        # 3. Feed Forward
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Norms (One for each sub-layer)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, 
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None):
        """
        x: Decoder input (Batch, Tgt_Seq, d_model)
        enc_out: Encoder output (Batch, Src_Seq, d_model)
        src_mask: Mask for cross-attention (hiding padding in source)
        tgt_mask: Mask for self-attention (hiding future tokens + padding in target)
        """
        
        # 1. Masked Self-Attention Sublayer
        # We pass x as Q, K, and V
        attn_out, _ = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 2. Cross-Attention Sublayer
        # Query = x (Decoder state)
        # Key, Value = enc_out (Encoder output)
        attn_out, _ = self.cross_attn(q=x, k=enc_out, v=enc_out, mask=src_mask)
        x = self.norm2(x + self.dropout(attn_out))
        
        # 3. Feed Forward Sublayer
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        
        return x
    
class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, N: int, heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        
        # Create N stacked layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, heads, d_ff, dropout) 
            for _ in range(N)
        ])
        
        self.norm = nn.LayerNorm(d_model) # Optional: Final normalization

    def forward(self, src: torch.Tensor, mask: torch.Tensor = None):
        x = self.embed(src)
        x = self.pe(x)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, N: int, heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, heads, d_ff, dropout) 
            for _ in range(N)
        ])
        
        self.norm = nn.LayerNorm(d_model) # Optional: Final normalization

    def forward(self, trg: torch.Tensor, enc_out: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        x = self.embed(trg)
        x = self.pe(x)
        
        for layer in self.layers:
            # We pass the encoder output to every decoder layer for cross-attention
            x = layer(x, enc_out, src_mask, tgt_mask)
            
        return self.norm(x)
    
class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab_size: int, 
                 trg_vocab_size: int, 
                 src_pad_idx: int, 
                 trg_pad_idx: int, 
                 d_model: int = 512, 
                 N: int = 6, 
                 heads: int = 8, 
                 d_ff: int = 2048, 
                 dropout: float = 0.1,
                 device: torch.device = device):
        super().__init__()
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
        # 1. The Encoder Stack
        self.encoder = Encoder(src_vocab_size, d_model, N, heads, d_ff, dropout)
        
        # 2. The Decoder Stack
        self.decoder = Decoder(trg_vocab_size, d_model, N, heads, d_ff, dropout)
        
        # 3. Final Output Projection
        # Maps d_model vector to vocabulary probability distribution (logits)
        self.out_linear = nn.Linear(d_model, trg_vocab_size)
        
        # Initialize weights (Optional but recommended in paper - Xavier Uniform)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        """
        Creates a mask for the source sequence to ignore padding tokens.
        Input: (Batch, Src_Seq)
        Output: (Batch, 1, 1, Src_Seq) - shape allows broadcasting
        """
        # (Batch, 1, 1, Src_Seq)
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg):
        """
        Creates a combined mask for the target sequence:
        1. Padding Mask: Ignore padding tokens.
        2. Look-ahead Mask: Ignore future tokens (causal).
        """
        N, trg_len = trg.shape
        
        # 1. Padding Mask
        # (Batch, 1, 1, Trg_Seq)
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        # 2. Look-ahead Mask (Lower triangular matrix)
        # (Trg_Seq, Trg_Seq)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        
        # Combine: Must be non-padding AND non-future
        # Bitwise AND (&)
        trg_mask = trg_pad_mask & trg_sub_mask
        
        return trg_mask

    def forward(self, src, trg):
        """
        src: (Batch, Src_Seq)
        trg: (Batch, Trg_Seq)
        """
        # 1. Generate Masks
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        # 2. Run Encoder
        # Output: (Batch, Src_Seq, d_model)
        enc_out = self.encoder(src, src_mask)
        
        # 3. Run Decoder
        # Output: (Batch, Trg_Seq, d_model)
        dec_out = self.decoder(trg, enc_out, src_mask, trg_mask)
        
        # 4. Project to Vocabulary
        # Output: (Batch, Trg_Seq, Trg_Vocab_Size)
        output = self.out_linear(dec_out)
        
        return output