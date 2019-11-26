import torch
import numpy

input = torch.Tensor([1., 1., 1., 1., 1., 1.,-1])


class MusicMultiheadAttention(torch.nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, 
                 add_zero_attn=False, kdim=None, vdim=None):

        torch.nn.MultiheadAttention.__init__(self, embed_dim, num_heads,
                                             dropout=0., bias=True,
                                             add_bias_kv=False, add_zero_attn=False,
                                             kdim=None, vdim=None)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):

        # learning a separate relative position embedding Er of shape (H, L, Dh)
        Er = torch.randn([self.num_heads, query.size(1), self.head_dim], requires_grad=False)
        # we transpose the two last dimensions of Er to realize Q*Er^T 
        QEr = torch.matmul(Q, torch.transpose(Er,1,2))
        # QEr of shape (B, H, L, L)     
        
        # 1. Pad a dummy column vector of length L before the leftmost column.
        QEr = torch.nn.functional.pad(QEr, (1,0), mode="constant", value=0)

        # 2. Reshape the matrix to have shape (L+1, L). 
        QEr = torch.reshape(QEr, [QEr.size(0), QEr.size(1), QEr.size(3), QEr.size(2)])
        
        # 3. Slice that matrix to retain only the last l rows and all the columns, 
        # resulting in a (L, L) matrix again, but now absolute-by-absolute indexed, 
        # which is the S rel that we need.
        S_rel = QEr[:,:,1:,:]
        
        return None

class MusicTransformerEncoderLayer(torch.nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048,
                 dropout=0.1, activation="relu"):
                 torch.nn.TransformerEncoderLayer.__init__(self, d_model,nhead)
                 # OverRide
                 self.self_attn = MusicMultiheadAttention(d_model, nhead)

class MusicTransformerDecoderLayer(torch.nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048,
                 dropout=0.1, activation="relu"):
                 torch.nn.TransformerDecoderLayer.__init__(self, d_model,nhead)
                 # OverRide
                 self.self_attn = MusicMultiheadAttention(d_model, nhead)

bob = torch.nn.Transformer(custom_encoder=MusicTransformerEncoderLayer(512, 8),
                     custom_decoder=MusicTransformerDecoderLayer(512, 8))
print(bob)
#print(square_tensor(input))
