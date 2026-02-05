import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        in_dim (int): Size of embedding dim of input
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        in_dim: int,
        E_total: int,
        nheads: int,
        dropout: float,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads = nheads
        self.dropout = dropout
        self.packed_proj = nn.Linear(in_dim, E_total * 3, bias=bias, **factory_kwargs)
        E_out = in_dim
        self.out_proj = nn.Linear(E_total, E_out, bias=bias, **factory_kwargs)
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads
        self.bias = bias

    def forward(
        self,
        query: torch.Tensor,
        # attn_mask=None,
        is_causal=False,
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (``N``, ``L_q``, ``E_qk``)
            key (torch.Tensor): key of shape (``N``, ``L_kv``, ``E_qk``)
            value (torch.Tensor): value of shape (``N``, ``L_kv``, ``E_v``)
            attn_mask (torch.Tensor, optional): attention mask of shape (``N``, ``L_q``, ``L_kv``) to pass to SDPA. Default: None
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        # Step 1. Apply input projection
        result = self.packed_proj(query)
        query, key, value = torch.chunk(result, 3, dim=-1)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        # Step 3. Run SDPA
        # (N, nheads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout, is_causal=is_causal
        )
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output


class GPTFeedForward(nn.Module):
    def __init__(self, cfg: dict, approximate="tanh"):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"])
        self.fc2 = nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        self.gelu = nn.GELU(approximate=approximate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class GPTTransformerBlock(nn.Module):
    def __init__(self, cfg: dict, approximate="tanh"):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg["emb_dim"])
        self.ln2 = nn.LayerNorm(cfg["emb_dim"])
        self.mha = MultiHeadSelfAttention(
            in_dim=cfg["emb_dim"],
            E_total=cfg["emb_dim"],
            nheads=cfg["n_heads"],
            dropout=cfg["dropout_rate"],
            bias=cfg["qkv_bias"],
        )
        self.ffn = GPTFeedForward(cfg, approximate=approximate)
        self.dropout = nn.Dropout(cfg["dropout_rate"])

    def forward(self, x):
        # Multi-head attention with residual connection
        attn_output = self.mha(self.ln1(x), is_causal=True)
        x = x + self.dropout(attn_output)

        # Feed-forward network with residual connection
        ffn_output = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_output)

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg: dict, approximate="tanh"):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.position_embedding = nn.Embedding(cfg["max_seq_len"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["dropout_rate"])
        self.blocks = nn.ModuleList(
            [
                GPTTransformerBlock(cfg, approximate=approximate)
                for _ in range(cfg["n_layers"])
            ]
        )
        self.ln_f = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        # Weight tying
        self.out_head.weight = self.token_embedding.weight

    def forward(self, input_ids):
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, device=input_ids.device)

        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.out_head(x)

        return logits
