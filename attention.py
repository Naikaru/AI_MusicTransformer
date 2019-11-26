import torch
import numpy

input = torch.Tensor([1., 1., 1., 1., 1., 1.,-1])


def square_tensor(input, fill=0.5):
    l = len(input)
    print(l)
    tensor = torch.ones((l,l))
    print(tensor.size())
    t = tensor.new_full((l,l), fill)
    return t


#class MusicMultiheadAttention(torch.nn.Module):
#    def __init__(self, d_model, nhead, dropout=0.1):
#        torch.nn.Module.__init__(self)

class MusicMultiheadAttention(torch.nn.MultiheadAttention):
    def __init__(self, d_model, nhead, dropout=0.1):
        torch.nn.MultiheadAttention.__init__(self, embed_dim, num_heads,
                                             dropout=0., bias=True,
                                             add_bias_kv=False, add_zero_attn=False,
                                             kdim=None, vdim=None)

    def forward():
        # Implement forward method.
        return None

class MusicTransformerEncoderLayer(torch.nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048,
                 dropout=0.1, activation="relu"):
                 torch.nn.TransformerEncoderLayer.__init__(self, d_model,nhead)
                 # OVERRIIIIIIDE
                 self.self_attn = MusicMultiheadAttention(d_model, nhead)

class MusicTransformerDecoderLayer(torch.nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048,
                 dropout=0.1, activation="relu"):
                 torch.nn.TransformerDecoderLayer.__init__(self, d_model,nhead)
                 # OVERRIIIIIIDE
                 self.self_attn = MusicMultiheadAttention(d_model, nhead)

bob = torch.nn.Transformer(custom_encoder=MusicTransformerEncoderLayer(512, 8),
                     custom_decoder=MusicTransformerDecoderLayer(512, 8))
print(bob)
#print(square_tensor(input))
