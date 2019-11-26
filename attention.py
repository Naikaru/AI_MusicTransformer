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
    def __init__(self, embed_dim, nhead, dropout=0.1, bias=True, add_bias_kv=False, 
                 add_zero_attn=False, kdim=None, vdim=None):
        torch.nn.MultiheadAttention.__init__(self, embed_dim, nhead, dropout=0.1, 
                                             bias=True, add_bias_kv=False, 
                                             add_zero_attn=False, kdim=None, vdim=None)

    def forward(self, query, key, value):
        Q, K, V = self.transform_input(query, key, value)
        # Reshaping the matrices 
        # Each L × D query, key, and value matrix is then split into H L × D 
        # h_D parts or attention heads, indexed by h, and with dimension D_h = D/H
        Q = self.matrix_to_heads(Q)
        K   = self.matrix_to_heads(K)
        V = self.matrix_to_heads(V)
        
        z_attention = attention(Q, K, V)
        
        return None

    def attention(weights_q, weights_k, weights_v):
        return None
    
    def matrix_to_heads(self, qkv):
        '''
            Takes a query/key/value (qkv) matrix and reshapes it to  H LxD_h heads 
            with dimension D_h = D/H
        '''
        batch_size_q = qkv.size(0)
        #print(qkv.size())
        qkv = torch.reshape(qkv, (batch_size_q, qkv.size(0), self.num_heads, self.head_dim))
        #print(self.head_dim // self.num_heads)
        #print(qkv.size())
        return qkv

    def transform_input(self, query, key, value):
        '''
            Transforming the input vector, X, of LxD dimension 
            into 
                queries: Q = XW^Q 
                keys:    K = XW^K
            and values:  V = XW^V
            which are all DxD square matrices.
        '''
        l_query = query.size(1)
        weights_q = torch.nn.Linear(l_query, l_query)
        
        #print(L_q)
        l_key = key.size(1)
        weights_k = torch.nn.Linear(l_key, l_key)
        
        l_value = key.size(1)
        weights_v = torch.nn.Linear(l_value, l_value)
        
        return weights_q(query), weights_k(key), weights_v(value)
    

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

#bob = torch.nn.Transformer(custom_encoder=MusicTransformerEncoderLayer(512, 8),
#                     custom_decoder=MusicTransformerDecoderLayer(512, 8))
#print(bob)
#print(square_tensor(input))
tensor = torch.ones((4,4,512))
#print(tensor)

mma = MusicMultiheadAttention(512, 8) 
#print(mma)

mma.forward(tensor, tensor, tensor)