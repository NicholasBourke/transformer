import math

import torch
import torch.nn as nn
from torch.nn.functional import relu, log_softmax



def positional_encoding(X):
    n = X.size(0)
    d = X.size(1)

    positions = torch.arange(n) + 1
    factors = torch.exp(torch.arange(0, d, 2) / d * -math.log(10000.0))
    terms = torch.outer(positions, factors)

    P = torch.zeros((n, d))
    P[:, 0::2] = torch.sin(terms)
    P[:, 1::2] = torch.cos(terms)

    return X + P


class FeedForwardSubLayer(nn.Module):
    def __init__(self, d_mod, d_ff):
        super(FeedForwardSubLayer, self).__init__()
        self.W1 = nn.Linear(d_mod, d_ff)
        self.W2 = nn.Linear(d_ff, d_mod)

    def forward(self, X):
        # input x size=[n, d_mod]
        X = relu(self.W1(X))    # size=[n, d_ff]
        X = self.W2(X)          # size=[n, d_mod]
        return X


class MultiHeadAttention(nn.Module):
    def __init__(self, d_mod, d_k, d_v, n_heads, masked=False):
        super(MultiHeadAttention, self).__init__()
        self.WQ = nn.ModuleList([nn.Linear(d_mod, d_k) for i in range(n_heads)])
        self.WK = nn.ModuleList([nn.Linear(d_mod, d_k) for i in range(n_heads)])
        self.WV = nn.ModuleList([nn.Linear(d_mod, d_v) for i in range(n_heads)])
        self.WO = nn.Linear(n_heads*d_v, d_mod)
        self.n_heads = n_heads
        self.masked = masked

    def forward(self, Q_input, K_input, V_input):
        head_attns = []
        for i in range(self.n_heads):
            query = self.WQ[i](Q_input)
            key = self.WK[i](K_input)
            value = self.WV[i](V_input)
            attn = self.scaled_dot_product_attention(query, key, value)
            head_attns.append(attn)
        Z = torch.cat(head_attns, dim=1)
        multi_attn = self.WO(Z)
        return multi_attn

    def scaled_dot_product_attention(self, query, key, value):
        # query size=[n, d_k]
        # key   size=[n, d_k]
        # value size=[n, d_v]
        # so far only applies to Encoder self-attention
        d_k = query.size(-1)
        query_key = torch.matmul(query, key.transpose(-2, -1))  # size=[n, n]
        if self.masked:
            query_key = torch.tril(query_key)
        scores = log_softmax(query_key / math.sqrt(d_k), dim=0) # size=[n, n]
        attention = torch.matmul(scores, value)                 # size=[n, d_v]

        return attention




class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, input_dim):
        super(LayerNorm, self).__init__()
        self.gain = nn.Parameter(torch.ones(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        # self.eps = eps

    def forward(self, input):
        mean = input.mean(-1, keepdim=True)
        std = input.std(-1, keepdim=True)
        return self.gain / std * (input - mean) + self.bias





class EncoderLayer(nn.Module):
    def __init__(self, n, d_mod, self_attn_sl, feedforward_sl):
        super(EncoderLayer, self).__init__()
        self.d_mod = d_mod
        self.attention = self_attn_sl
        self.norm_1 = LayerNorm((n, d_mod))
        self.feedforward = feedforward_sl
        self.norm_2 = LayerNorm((n, d_mod))

    def forward(self, X):
        Z = self.attention(X, X, X)
        Z = self.norm_1(X + Z)
        R = self.feedforward(Z)
        R = self.norm_2(Z + R)
        return R

class EncoderStack(nn.Module):
    def __init__(self, vocab_size, n, d_mod, d_ff, d_k, d_v, n_heads=8, n_layers=6):
        super(EncoderStack, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_mod)
        self.stack = nn.Sequential()
        for k in range(n_layers):
            self_attn_sublayer = MultiHeadAttention(d_mod, d_k, d_v, n_heads)
            feedforward_sublayer = FeedForwardSubLayer(d_mod, d_ff)
            layer = EncoderLayer(n, d_mod, self_attn_sublayer, feedforward_sublayer)
            self.stack.append(layer)

    def forward(self, X):
        X = self.embedding(X)
        X = positional_encoding(X)
        memory = self.stack(X)
        return memory
        


class DecoderLayer(nn.Module):
    def __init__(self, memory, m, d_mod, self_attn_sl, encdec_attn_sl, feedforward_sl):
        super(DecoderLayer, self).__init__()
        self.memory = memory
        self.d_mod = d_mod
        self.self_attention = self_attn_sl
        self.norm_1 = LayerNorm((m, d_mod))
        self.encdec_attention = encdec_attn_sl
        self.norm_2 = LayerNorm((m, d_mod))
        self.feedforward = feedforward_sl
        self.norm_3 = LayerNorm((m, d_mod))

    def forward(self, Y):
        self_attn = self.self_attention(Y, Y, Y)
        self_attn = self.norm_1(Y + self_attn)
        ed_attn = self.encdec_attention(self_attn, self.memory, self.memory)
        ed_attn = self.norm_2(self_attn + ed_attn)
        R = self.feedforward(ed_attn)
        R = self.norm_3(ed_attn + R)
        return R



class DecoderStack(nn.Module):
    def __init__(self, memory, vocab_size, n, d_mod, d_ff, d_k, d_v, n_heads=8, n_layers=6):
        super(DecoderStack, self).__init__()
        self.memory = memory
        self.embedding = nn.Embedding(vocab_size, d_mod)
        self.stack = nn.Sequential()
        for k in range(n_layers):
            self_attn_sublayer = MultiHeadAttention(d_mod, d_k, d_v, n_heads, masked=True)
            encdec_attn_sublayer = MultiHeadAttention(d_mod, d_k, d_v, n_heads)
            feedforward_sublayer = FeedForwardSubLayer(d_mod, d_ff)
            layer = DecoderLayer(memory, n, d_mod, self_attn_sublayer, encdec_attn_sublayer, feedforward_sublayer)
            self.stack.append(layer)
        self.probability = nn.Linear(d_mod, vocab_size)

    def forward(self, Y):
        Y = self.embedding(Y)
        Y = positional_encoding(Y)
        output = self.stack(Y)
        prob_dist = log_softmax(self.probability(output), dim=1)
        return prob_dist







