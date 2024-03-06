import torch
from torch.nn.functional import relu, log_softmax

from layers import EncoderStack, DecoderStack






src_vocab_size = 1000
tgt_vocab_size = 800
n = 8
m = 10
d_mod = 512
d_ff = 2048
d_k = 64
d_v = 64

X = torch.randint(src_vocab_size, (n, 1)).squeeze()
Y = torch.randint(tgt_vocab_size, (m, 1)).squeeze()


encoder = EncoderStack(src_vocab_size, n, d_mod, d_ff, d_k, d_v)
memory = encoder(X)

decoder = DecoderStack(memory, tgt_vocab_size, m, d_mod, d_ff, d_k, d_v)
prob_dist = decoder(Y)

print(prob_dist.size())
sum_check = torch.sum(torch.exp(prob_dist), dim=1)
print(sum_check)