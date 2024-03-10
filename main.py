import torch
from torch.nn.functional import one_hot, relu, log_softmax

from layers import Embed, EncoderStack, DecoderStack




src_vocab_size = 1000
tgt_vocab_size = 800
d_vocab = src_vocab_size + tgt_vocab_size
n = 8
m = 10
d_mod = 512
d_ff = 2048
d_k = 64
d_v = 64


X = torch.randint(src_vocab_size, (n, 1)).squeeze()
Y = torch.randint(tgt_vocab_size, (m, 1)).squeeze()

X_token = one_hot(X, num_classes=d_vocab).type(torch.float)
Y_token = one_hot(Y, num_classes=d_vocab).type(torch.float)


embed_layer = Embed(d_vocab, d_mod)

encoder = EncoderStack(embed_layer, d_vocab, n, d_mod, d_ff, d_k, d_v)
memory = encoder(X_token)

decoder = DecoderStack(embed_layer, memory, d_vocab, m, d_mod, d_ff, d_k, d_v)
prob_dist = decoder(Y_token)

print(prob_dist.size())
sum_check = torch.sum(torch.exp(prob_dist), dim=1)
print(sum_check)