import torch
import torch.nn.functional as F
import numpy

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
        
        z_attention = self.attention(Q, K, V)
        print(z_attention.size())
        
        return None

    def attention(self, Q, K, V):
        # Dh = self.head_dim // self.num_heads
        # Add S^rel for correct implementation.
        QK = torch.matmul(Q, torch.transpose(K, 2, 3))
        QK_div = QK / (numpy.sqrt((self.head_dim // self.num_heads)))
        activation = F.softmax(QK_div, -1)
        attention = torch.matmul(activation, V)
        return attention
    
    def matrix_to_heads(self, qkv):
        '''
            Takes a query/key/value (qkv) matrix and reshapes it to  H LxD_h heads 
            with dimension D_h = D/H
        '''
        batch_size_q = qkv.size(0)
        #print(qkv.size())
        #qkv = torch.reshape(qkv, (batch_size_q, qkv.size(0), self.num_heads, self.head_dim))
        qkv = torch.reshape(qkv, (batch_size_q, self.num_heads, qkv.size(1), self.head_dim))
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
        l_query = query.size(2)
        print(l_query)
        weights_q = torch.nn.Linear(l_query, l_query)
        
        #print(L_q)
        l_key = key.size(2)
        weights_k = torch.nn.Linear(l_key, l_key)
        
        l_value = key.size(2)
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

tensor = torch.ones((4,4,512))
#print(tensor)

mma = MusicMultiheadAttention(512, 8) 
#print(mma)

mma.forward(tensor, tensor, tensor)
#mma.attention(1,2,3)