from logsable.common_imports import *



class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T, :]

# ----- Simple Transformer Encoder classifier -----
class NeuralLogClassifier(nn.Module):
    def __init__(self, num_tokens: int, d_model: int = 128, nhead: int = 8, num_layers: int = 2, dim_ff: int = 512, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.emb = nn.Embedding(num_tokens, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.cls = nn.Linear(d_model, 2)

    def forward(self, x_ids):
        # x_ids: [B, T] integer EventId indices
        h = self.emb(x_ids)          # [B, T, d_model]
        h = self.pos(h)
        h = self.enc(h)
        h = self.norm(h)
        h = h.mean(dim=1)            # mean pool over time
        return self.cls(h)


class DeepLogClassifier(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        embed_dim: int = 64,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_tokens   = num_tokens
        self.embed_dim    = embed_dim
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers

        self.embedding = nn.Embedding(num_embeddings=num_tokens, embedding_dim=embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_size, 2)  # 2 classes: normal / anomaly

    def forward(self, x):
        """
        x: LongTensor of shape (batch, seq_len) with token IDs.
        Returns: logits of shape (batch, 2)
        """
        # (B, T) -> (B, T, E)
        emb = self.embedding(x)
        # (B, T, E) -> (B, T, H)
        out, (h_n, c_n) = self.lstm(emb)
        # last hidden state of last layer: (num_layers, B, H) -> (B, H)
        h_last = h_n[-1]  # shape (B, H)
        h_last = self.dropout(h_last)
        logits = self.fc(h_last)
        return logits


class LogAnomalyClassifier(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        embed_dim: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(num_embeddings=num_tokens, embedding_dim=embed_dim, padding_idx=0)
        self.lstm_seq = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.count_proj = nn.Linear(num_tokens, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 2)

    def forward(self, x):
        """
        x: LongTensor (B, T) event IDs.
        Returns: logits (B, 2)
        """
        B, T = x.shape
        emb = self.embedding(x)
        out_seq, (h_n, _) = self.lstm_seq(emb)
        h_seq = h_n[-1]

        idx = x.clamp(0, self.num_tokens - 1)
        # O(B*T) count vector — avoid materializing one_hot(B, T, num_tokens)
        count = torch.zeros(B, self.num_tokens, device=x.device, dtype=emb.dtype)
        count.scatter_add_(1, idx, torch.ones_like(idx, dtype=count.dtype))
        h_quant = self.count_proj(count)

        h = torch.cat([h_seq, h_quant], dim=1)
        h = self.dropout(h)
        return self.fc(h)


class LogBERTClassifier(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(num_tokens, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos = PositionalEncoding(d_model, max_len=max_len + 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.cls_head = nn.Linear(d_model, 2)

    def forward(self, x_ids):
        # x_ids: [B, T] integer event indices
        B, T = x_ids.shape
        # Shift token ids if we use 0 for padding; optional: keep 0..num_tokens-1 as-is
        h = self.emb(x_ids)  # [B, T, d_model]
        cls_expanded = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        h = torch.cat([cls_expanded, h], dim=1)  # [B, 1+T, d_model]
        h = self.pos(h)
        h = self.encoder(h)
        h = self.norm(h)
        cls_out = h[:, 0, :]  # [B, d_model]
        return self.cls_head(cls_out)
