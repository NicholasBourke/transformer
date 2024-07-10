import math
import torch
import torch.nn as nn
import torch.nn.functional as F



def positional_encoding(x):
    n, d = x.size()

    positions = torch.arange(n) + 1
    factors = torch.exp(torch.arange(0, d, 2) / d * -math.log(10000.0))
    terms = torch.outer(positions, factors)

    P = torch.zeros((n, d))
    P[:, 0::2] = torch.sin(terms)
    P[:, 1::2] = torch.cos(terms)

    return x + P


class Embed(nn.Module):
    def __init__(self, d_vocab, d_mod):
        super(Embed, self).__init__()
        a = math.sqrt(1/d_vocab)
        self.embed = nn.Parameter(2*a * torch.rand(d_vocab, d_mod) - a)
        self.d_vocab = d_vocab
        self.d_mod = d_mod

    def forward(self, X):
        if X.size(-1) == self.d_vocab:
            X_emb = torch.matmul(X, self.embed) * math.sqrt(self.d_mod)
        if X.size(-1) == self.d_mod:
            X_emb = torch.matmul(X, self.embed.transpose(1, 0))
        return X_emb


class LayerNorm(nn.Module):
    def __init__(self, norm_dim, eps=1e-05):
        super(LayerNorm, self).__init__()
        self.gain = nn.Parameter(torch.ones(norm_dim))
        self.bias = nn.Parameter(torch.zeros(norm_dim))
        self.eps = eps
        self.num_dims = len(norm_dim)

    def forward(self, x):
        dims = tuple(range(-self.num_dims, 0))
        mu = x.mean(dims, keepdim=True)
        var = x.var(dims, keepdim=True)
        return (x - mu) / torch.sqrt(var + self.eps) * self.gain + self.bias



class MultiHeadAttention(nn.Module):
    def __init__(self, cfg, masked=False):
        super(MultiHeadAttention, self).__init__()
        assert cfg.D % cfg.n_head == 0
        self.cfg = cfg
        self.masked = masked

        self.Wqkv = nn.Linear(cfg.D, 3*cfg.D, bias=cfg.bias)
        self.Wo = nn.Linear(cfg.D, cfg.D, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        q, k, v = self.Wqkv(x).split(self.cfg.D, dim=-1)
        q = q.view(-1, self.cfg.L, self.cfg.n_head, self.cfg.D//self.cfg.n_head).transpose(1, 2)
        k = k.view(-1, self.cfg.L, self.cfg.n_head, self.cfg.D//self.cfg.n_head).transpose(1, 2)
        v = v.view(-1, self.cfg.L, self.cfg.n_head, self.cfg.D//self.cfg.n_head).transpose(1, 2)

        attn = self.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(x.size())
        return self.dropout(self.Wo(attn))

    def scaled_dot_product_attention(self, q, k, v):
        qk = q @ k.transpose(-2, -1)
        if self.masked:
            mask = torch.tril(torch.ones(self.cfg.L, self.cfg.L)).to(q.device, dtype=torch.int)
            qk.masked_fill_(mask == 0, float('-inf'))
        qk = F.softmax(qk / math.sqrt(k.size(-1)), dim=-1)
        return qk @ v


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super(FeedForward, self).__init__()
        self.W1 = nn.Linear(cfg.D, 4*cfg.D, bias=False)
        self.W2 = nn.Linear(4*cfg.D, cfg.D, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.W2(self.relu(self.W1(x))))



class EncoderLayer(nn.Module):
    def __init__(self, cfg):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(cfg)
        self.nrm_1 = LayerNorm((cfg.L, cfg.D))
        self.ff = FeedForward(cfg)
        self.nrm_2 = LayerNorm((cfg.L, cfg.D))

    def forward(self, x):
        a = self.nrm_1(x + self.dropout(self.attn(x)))
        c = self.nrm_2(a + self.dropout(self.ff(a)))
        return c


class EncoderStack(nn.Module):
    def __init__(self, cfg):
        super(EncoderStack, self).__init__()
        self.embed = nn.Embedding(cfg.K, cfg.D)
        self.stack = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_layer)])
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.embed(x)
        x = self.dropout(positional_encoding(x))
        for layer in self.stack:
            x = layer(x)
        return x
        


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super(DecoderLayer, self).__init__()
        self.attn_1 = MultiHeadAttention(cfg, masked=True)
        self.nrm_1 = LayerNorm((cfg.L, cfg.D))
        self.attn_2 = MultiHeadAttention(cfg)
        self.nrm_2 = LayerNorm((cfg.L, cfg.D))
        self.ff = FeedForward(cfg)
        self.nrm_3 = LayerNorm((cfg.L, cfg.D))

    def forward(self, y, c):
        a1 = self.nrm_1(y + self.attn_1(y, y, y))
        a2 = self.nrm_2(a1 + self.attn_2(a1, c, c))
        z = self.nrm_3(a2 + self.ff(a2))
        return z


class DecoderStack(nn.Module):
    def __init__(self, cfg):
        super(DecoderStack, self).__init__()
        self.embed = nn.Embedding(cfg.K, cfg.D)
        self.stack = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_layer)])
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, y, c):
        y = self.embed(y)
        y = self.dropout(positional_encoding(y))
        for layer in self.stack:
            y = layer(y, c)
        return y
    


class Transformer(nn.Module):
    def __init__(self, cfg):
        super(Transformer, self).__init__()
        self.encoder = EncoderStack(cfg)
        self.decoder = DecoderStack(cfg)
        self.linear = nn.Linear(cfg.D, cfg.K, bias=False)

    def forward(self, x, y):
        c = self.encoder(x)
        z = self.decoder(y, c)
        return F.softmax(self.linear(z))
