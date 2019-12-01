'''
    Was created in google collab and later put into this github so it is ugly... Sorry about that! 
'''

from google.colab import drive

drive.mount('/gdrive')
gdrive_root = '/gdrive/My Drive'
gdrive_data = '/gdrive/My Drive/my_data'

#!nvcc --version

import pickle
import os
import sys
import math
import random
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

##########################
Hyperparameters
##########################

d_model = 512
nhead = 8
dim_feedforward = 1024
dropout = 0.2
num_layer = 6
batch_size = 8
sequence_length = 1024
warmup_steps = 4000
pad_token = 1   
# vocabulary_size = 388 # depends on the dataset

##########################
Encoder function
##########################

#Import library for encoder function

#!ls /gdrive/My\ Drive/my_data/library/processor.py
#!cat '/gdrive/My Drive/my_data/library/processor.py'

sys.path.append('/gdrive/My Drive/my_data/library')

from processor import encode_midi

##########################
encode midi
##########################
def encode_midi_files(midi_dir_path, save_dir_path, extension):
  #create directory for saving files
  os.makedirs(save_dir_path, exist_ok=True)
  #get all midi files from midi_directory
  for file in os.listdir(midi_dir_path):
    if file.endswith(tuple(extension)):
        print(os.path.join(midi_dir_path, file))
        print(file + ' is being processed', flush=True)
        try:
          encoded_file = encode_midi(midi_dir_path+file)
        except KeyboardInterrupt:
            print(' Stopped by keyboard')
            return
        except EOFError:
            print('EOF Error')
            return
        with open(save_dir_path+file+'.encoded', 'wb') as f:
            pickle.dump(encoded_file, f)

midi_dir_path = '/gdrive/My Drive/my_data/midi_files/'
save_dir_path = '/gdrive/My Drive/my_data/encoded_midi/'
train_dir_path = '/gdrive/My Drive/my_data/training_set/'
test_dir_path = '/gdrive/My Drive/my_data/test_set/'
extension = ['.mid', '.midi']

import shutil

'''
    Gets the files from a directory to which midi datasets have been downloaded
'''

def create_dataset(save_dir_path, split_ratio=0.9):
    dataset = [file for file in os.listdir(save_dir_path)]
    np.random.shuffle(dataset)

    train_set = dataset[:int(len(dataset) * split_ratio)]    
    test_set = dataset[int(len(dataset) * split_ratio):]
    
    shutil.rmtree(train_dir_path)
    shutil.rmtree(test_dir_path)

    os.makedirs(train_dir_path, exist_ok=True)
    os.makedirs(test_dir_path, exist_ok=True)

    for file in os.listdir(save_dir_path):
        if os.stat(save_dir_path+file).st_size != 0:
            if file in test_set:
                shutil.copyfile(save_dir_path+file, test_dir_path+file)
            else:
                shutil.copyfile(save_dir_path+file, train_dir_path+file)

def load_dataset(train_dir_path, test_dir_path):
    #load all encoded file
    train_set = [file for file in os.listdir(train_dir_path)]
    test_set = [file for file in os.listdir(test_dir_path)]

    return train_set, test_set

def generate_batch(dataset, dir_path, sequence_length=1024, batch_size=8):
    sequence_length += 1
    batch_midi = []
    while len(batch_midi) < batch_size:
        file = random.choice(dataset)
        #if the midi contains more sequence that the sequence length
        try:
            with open(dir_path+file, 'rb') as f:
                data = pickle.load(f)
        except:
            print(dir_path+file + " file not found.")
        if sequence_length <= len(data):
            begin_index = random.randrange(0, len(data) - sequence_length)
            data = data[begin_index:begin_index + sequence_length]
            batch_midi.append(data)
    batch_midi = torch.Tensor(batch_midi)
    inputs = batch_midi[:, :-1]
    labels = batch_midi[:, 1:]
    return inputs, labels
    
##########################
Attention code 
##########################

class DynamicPositionEmbedding(torch.nn.Module):
    
    # From github https://github.com/jason9693/midi-processor.git 
    def __init__(self, embedding_dim, max_seq=1024):
        super().__init__()
        embed_sinusoid_list = np.array([[
            [
                math.sin(
                    pos * math.exp(-math.log(10000) * i/embedding_dim) *
                    math.exp(math.log(10000)/embedding_dim * (i % 2)) + 0.5 * math.pi * (i % 2)
                )
                for i in range(embedding_dim)
            ]
            for pos in range(max_seq)
        ]])
        self.positional_embedding = embed_sinusoid_list

    def forward(self, x):
        x = x + torch.from_numpy(self.positional_embedding[:, :x.size(1), :]).to(x.device, dtype=x.dtype)
        return x

class MusicMultiheadAttention(torch.nn.MultiheadAttention):
    def __init__(self, embed_dim, nhead, dropout=0.1, bias=True, add_bias_kv=False, 
                 add_zero_attn=False, kdim=None, vdim=None):
        
        torch.nn.MultiheadAttention.__init__(self, embed_dim, nhead, dropout=0.1, 
                                             bias=True, add_bias_kv=False, 
                                             add_zero_attn=False, kdim=None, vdim=None)
        
        self.embed_dim = embed_dim
        self.weights_q = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.weights_k = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.weights_v = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.weights_o = torch.nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        Q, K, V = self.transform_input(query, key, value)
        # Reshaping the matrices 
        # Each L × D query, key, and value matrix is then split into H L × D 
        # h_D parts or attention heads, indexed by h, and with dimension D_h = D/H
        Q = self.matrix_to_heads(Q)
        K = self.matrix_to_heads(K)
        V = self.matrix_to_heads(V)
        
        # learning a separate relative position embedding Er of shape (H, L, Dh)
        Er = torch.randn([self.num_heads, query.size(1), self.head_dim], requires_grad=False).to(device)

        # we transpose the two last dimensions of Er to realize Q*Er^T 
        QEr = torch.matmul(Q, torch.transpose(Er,1,2))
        # QEr of shape (B, H, L, L)     
        # QEr = torch.einsum('bhld,ld->bhll', [Q, Er])

        # 1. Pad a dummy column vector of length L before the leftmost column.
        QEr = torch.nn.functional.pad(QEr, (1,0), mode="constant", value=0)

        # 2. Reshape the matrix to have shape (L+1, L). 
        QEr = torch.reshape(QEr, [QEr.size(0), QEr.size(1), QEr.size(3), QEr.size(2)])
        
        # 3. Slice that matrix to retain only the last l rows and all the columns, 
        # resulting in a (L, L) matrix again, but now absolute-by-absolute indexed, 
        # which is the S rel that we need.
        S_rel = QEr[:,:,1:,:]

        z_attention = self.attention(Q, K, V, S_rel, attn_mask)
        z_attention = self.weights_o(z_attention)
        # Masking can be added and Dropout ?

        return z_attention

    def attention(self, Q, K, V, S, mask):
        # Dh = self.head_dim // self.num_heads
        logits = torch.add(torch.matmul(Q, torch.transpose(K, 2, 3)), S) / math.sqrt((self.head_dim // self.num_heads))
        # print("logits : ", logits.size())
        # print("mask : ", mask.size())
        if mask is not None:
        #    mask = mask.unsqueeze(1) #shape of mask must be broadcastable with shape of underlying tensor
            logits = logits.masked_fill(mask == 0, -1e9) #masked_fill fills elements of scores with -1e9 where mask == 0
        #if mask is not None:
        #    logits += (mask.to(torch.int64) * -1e9).to(logits.dtype)        
            
        activation = F.softmax(logits, -1)
        attention = torch.matmul(activation, V)
        attention = torch.reshape(attention, (attention.size(0), -1, self.embed_dim))
        return attention
    
    def matrix_to_heads(self, qkv):
        '''
            Takes a query/key/value (qkv) matrix and reshapes it to  B * H * L * D_h heads 
            with dimension D_h = D/H
        '''
        batch_size_q = qkv.size(0)
        #qkv = torch.reshape(qkv, (batch_size_q, qkv.size(0), self.num_heads, self.head_dim))
        qkv = torch.reshape(qkv, (batch_size_q, self.num_heads, qkv.size(1), self.head_dim))
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
        return self.weights_q(query), self.weights_k(key), self.weights_v(value)
    

class MusicTransformerEncoderLayer(torch.nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048,
                 dropout=0.1, activation="relu"):
        torch.nn.TransformerEncoderLayer.__init__(self, d_model, nhead)
        self.d_model = d_model
        # OverRide
        self.self_attn = MusicMultiheadAttention(d_model, nhead)

class MusicTransformerDecoderLayer(torch.nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048,
                 dropout=0.1, activation="relu"):
        torch.nn.TransformerDecoderLayer.__init__(self, d_model,nhead)
        self.d_model = d_model
        # OverRide
        self.self_attn = MusicMultiheadAttention(d_model, nhead)

class MusicTransformerEncoder(torch.nn.TransformerEncoder):
    def __init__(self, encoder_layer, vocabulary_size=390, num_encoder_layers=6, normalization=None):
        super().__init__(encoder_layer, num_encoder_layers, normalization)
        self.d_model = encoder_layer.d_model
        self.vocabulary_size = vocabulary_size
        self.dropout = torch.nn.Dropout(encoder_layer.dropout.p)
        self.embedding = torch.nn.Embedding(num_embeddings=self.vocabulary_size, embedding_dim=self.d_model)
    
    def forward(self, src, mask=None, src_key_padding_mask=None):
        
        pos_encoding = DynamicPositionEmbedding(self.d_model, src.size(1))
        
        src = math.sqrt(self.d_model) * self.embedding(src.to(torch.long).to(device))
        src = pos_encoding(src)
        src = self.dropout(src)

        return super().forward(src, mask=mask, src_key_padding_mask=src_key_padding_mask)


class MusicTransformerDecoder(torch.nn.TransformerDecoder):
    def __init__(self, decoder_layer, vocabulary_size=390, num_decoder_layers=6, normalization=None):
        super().__init__(decoder_layer, num_decoder_layers, normalization)
        self.d_model = decoder_layer.d_model
        self.vocabulary_size = vocabulary_size
        self.dropout = torch.nn.Dropout(decoder_layer.dropout.p)
        self.embedding = torch.nn.Embedding(num_embeddings=self.vocabulary_size, embedding_dim=self.d_model)

    def forward(self, tgt, memory, tgt_mask=None, 
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):        
                
        pos_encoding = DynamicPositionEmbedding(self.d_model, tgt.size(1))

        tgt = pos_encoding(math.sqrt(self.d_model) * self.embedding(tgt.to(torch.long).to(device)))
        tgt = self.dropout(tgt)

        return super().forward(tgt, memory, tgt_mask=tgt_mask, 
                memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask)


class MusicTransformer(torch.nn.modules.Transformer):
    def __init__(self, d_model=512, nhead=8, vocabulary_size=388,
                 num_encoder_layers=6, num_decoder_layers=6, 
                 dim_feedforward=2048, dropout=0.1, activation="relu", 
                 custom_encoder=None, custom_decoder=None):
        
        super().__init__(d_model=d_model, nhead=nhead, 
                         num_encoder_layers=num_encoder_layers,
                         num_decoder_layers=num_decoder_layers, 
                         dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, 
                         custom_encoder=custom_encoder, custom_decoder=custom_decoder)
        
        self.vocabulary_size = vocabulary_size
        ###        
        self.fc = torch.nn.Linear(self.d_model, self.vocabulary_size)
        self._reset_parameters()

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        #if src.size(1) != tgt.size(1):
        #    raise RuntimeError("the batch number of src and tgt must be equal")

        #if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
        #    raise RuntimeError("the feature number of src and tgt must be equal to d_model")
            
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)

        output = self.fc(output)        
        
        return output
    
##########################
Training
##########################

 
# vocabulary_size depends on the midi encodding
# ~> 388(+2) for encoded_midi / epiano compt 
# ~> 46(+2) for encoded_midi / epiano compt                 

# DEFINING THE MODEL

vocabulary_size = 390

normalization = torch.nn.LayerNorm(d_model)

custom_encoder_layer = MusicTransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                               dim_feedforward=dim_feedforward, 
                                               dropout=dropout, activation="relu")

custom_decoder_layer = MusicTransformerDecoderLayer(d_model=d_model, nhead=nhead, 
                                               dim_feedforward=dim_feedforward, 
                                               dropout=dropout, activation="relu")

custom_encoder = MusicTransformerEncoder(custom_encoder_layer, vocabulary_size, num_layer, normalization)
custom_decoder = MusicTransformerDecoder(custom_decoder_layer, vocabulary_size, num_layer, normalization)

model = MusicTransformer(d_model=d_model, nhead=nhead, 
                         vocabulary_size=vocabulary_size, 
                         num_encoder_layers=num_layer, 
                         num_decoder_layers=num_layer, 
                         dim_feedforward=dim_feedforward, 
                         dropout=dropout, activation="relu", 
                         custom_encoder=custom_encoder, 
                         custom_decoder=custom_decoder)
# Give model to the current device (hopefully cuda)
model.to(device)

# Optimizer
# Adam optimizer [20] with β 1 = 0.9, β 2 = 0.98 and  = 10 −9
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4)

# Define a scheduler to vary the learning rate

class Scheduler:
    def __init__(self, optimizer, d_model=d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.step_num = 1
        self.l_rate = 0
        self.warmup_steps = warmup_steps

    def step(self):        
        # increment step
        self.step_num += 1

        # compute new learning rate        
        self.l_rate = self.d_model**(-.5) * min(self.step_num**(-.5), self.step_num * self.warmup_steps**(-1.5))

        # update optimizer learning rate
        for p in optimizer.param_groups:
            p['lr'] = self.l_rate

        # update the weights in the network
        self.optimizer.step()
        
scheduler = Scheduler(optimizer, d_model, warmup_steps)

##########################
Load model if exists
##########################
train_data, test_data = load_dataset(train_dir_path, test_dir_path)

ckpt_dir = os.path.join(gdrive_root, '/my_data/library/checkpoints/')
if not os.path.exists(ckpt_dir):
  os.makedirs(ckpt_dir)
  
best_loss = 10.
#ckpt_path = os.path.join(ckpt_dir, 'MT-'+ str(datetime.datetime.now()) +'.pt')
#ckpt_path = os.path.join(ckpt_dir, 'MT.pt')
model_name = 'midi_encoded_6-1'
ckpt_path = '/gdrive/My Drive/my_data/library/checkpoints/train-nlayer_'+model_name+'.pt'
ckpt_backup_path = '/gdrive/My Drive/my_data/library/checkpoints/train-backup-nlayer_'+model_name+'.pt'
ckpt_fullbackup_path = '/gdrive/My Drive/my_data/library/checkpoints/train-fullbackup-nlayer_'+model_name+'.pt'
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path)
    try:
      model.load_state_dict(ckpt['my_model'])
      optimizer.load_state_dict(ckpt['optimizer'])
      best_acc = ckpt['best_loss']
    except RuntimeError as e:
        print('wrong checkpoint')
    else:    
      print('checkpoint is loaded !')
      print('current best loss : %.2f' % best_loss)

train_writer = SummaryWriter()
test_writer = SummaryWriter()


##########################
Training loop
##########################

# Training
max_epochs = 100
n_iter = 0

# each 50 iterations we are going to compare the losses
total_train_loss = []
total_valid_loss = []

for e in range(max_epochs):
    model.train()
    train_loss = []
    for b_train in range(len(train_data) // batch_size):
        
        n_iter += 1
        # train phase
        # feed data into the network and get outputs.
        # feed data into the network and get outputs.
        inputs, target = generate_batch(train_data, train_dir_path, sequence_length=sequence_length, batch_size=batch_size)
              
        # Train on GPU
        inputs.to(device)
        target.to(device)
        ys = target.contiguous().view(-1).to(torch.long).to(device)
        
        # Create mask for both input and target sequences
        input_mask, target_mask = create_masks(torch.reshape(inputs, (batch_size, 1, -1)), torch.reshape(target, (batch_size, 1, -1)), pad_token)        
        
        # feed data into the network and get outputs.
        preds_idx = model(inputs, target, input_mask, target_mask)
        
        # Flush out gradients computed at the previous step before computing gradients at the current step. 
        #       Otherwise, gradients would accumulate.
        optimizer.zero_grad()

        # calculate loss
        loss = F.cross_entropy(preds_idx.contiguous().view(preds_idx.size(-1), -1).transpose(0,1), ys, ignore_index = pad_token, size_average = False) / (count_nonpad_tokens(ys))

        # accumulates the gradient and backprogate loss.
        loss.backward()

        # performs a parameter update based on the current gradient
        scheduler.step()    
        
        print('\n====================================================')
        print('Epoch/Batch: {}/{}'.format(e, b_train))
        print('Train >>>> Loss: {:6.6}'.format(loss))

        train_loss.append(loss.item())

    print('\n**************************************************')
    print("\n*** Test *** ")
    
    # Validation step
    model.eval()
    valid_loss = []
    with torch.no_grad():
        for b_test in range(len(test_data) // batch_size):        
            inputs, target = generate_batch(test_data, test_dir_path, sequence_length=sequence_length, batch_size=batch_size)
                  
            # Train on GPU
            inputs.to(device)
            target.to(device)
            ys = target.contiguous().view(-1).to(torch.long).to(device)
            
            # Create mask for both input and target sequences
            input_mask, target_mask = create_masks(torch.reshape(inputs, (batch_size, 1,-1)), torch.reshape(target, (batch_size, 1, -1)), pad_token)        

            # Feed Forward
            preds_validate = model(inputs, target, input_mask, target_mask)
            loss = F.cross_entropy(preds_validate.contiguous().view(preds_validate.size(-1), -1).transpose(0,1), ys, \
                                    ignore_index = pad_token, size_average = False) / (count_nonpad_tokens(ys))
            valid_loss.append(loss.item())

    avg_train_loss = np.mean(train_loss)
    avg_valid_loss = np.mean(valid_loss)

    total_train_loss.append(avg_train_loss)
    total_valid_loss.append(avg_valid_loss)

    print("[Average Train Loss]: {:6.6}".format(avg_train_loss))
    print("[Average Testing Loss]: {:6.6}".format(avg_valid_loss))

    # save checkpoint whenever there is improvement in performance
    if avg_valid_loss < best_loss:
        best_loss = avg_valid_loss
        # Note: optimizer also has states ! don't forget to save them as well.
        ckpt = {'my_model':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'best_loss':best_loss}
        torch.save(ckpt, ckpt_path)
        print('checkpoint is saved !')

    train_writer.add_scalar('loss/train', avg_train_loss, global_step=n_iter)
    test_writer.add_scalar('loss/valid', avg_valid_loss, global_step=n_iter)
    print('\n**************************************************')
    # torch.cuda.empty_cache()
    # torch.save(model.state_dict(), '/gdrive/My Drive/my_data/library/checkpoints/train-{}.pt'.format(e))
    ckpt = {'my_model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'best_loss':best_loss}
    torch.save(model.state_dict(), ckpt_backup_path)
    torch.save(ckpt, ckpt_fullbackup_path)

train_writer = SummaryWriter()
test_writer = SummaryWriter()

##########################
Visualize
##########################

import matplotlib.pyplot as plt

plt.plot(total_train_loss, label='train loss')
plt.plot(total_valid_loss, label='validation loss')
plt.legend()

img_path = os.path.join('/gdrive/My Drive/my_data/library/checkpoints/', "losses_5-6.png")
plt.savefig(img_path)

##########################
Generate / Plot
##########################

model.eval()
inputs = torch.randint(0, 388, (1,, )).to(device)
gen_length = 300
generated_midi = torch.Tensor()
for seq in range(gen_length):
    if seq % 20 == 0: 
        print(seq)
    logits = F.softmax(model(inputs[:, :-1], inputs[:, 1:]), -1)

    logits = logits[0, :, :].to(device)
    
    one_hot = torch.distributions.OneHotCategorical(probs=logits[:,-1])
    res = one_hot.sample().argmax(-1).unsqueeze(-1).to(device)

    #res = torch.transpose(res, 0, 1)

    inputs = torch.cat((inputs[0], res), dim=-1)
    inputs = torch.reshape(inputs, (1, -1))

### Plot midi values vs index

from matplotlib import pyplot as pt

x = range(0, len(inputs[0]))
y = [ints.item() for ints in inputs[0]]

p = pt.scatter(x,noise, label='generated pitches')

pt.xlabel('index')
pt.ylabel('midi value (pitch)')
pt.show()
