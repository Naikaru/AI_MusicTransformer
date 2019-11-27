#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 22:05:47 2019

@author: quentin
"""
import torch
from attention import MusicTransformerEncoderLayer, MusicTransformerDecoderLayer

# Hyperparameters
d_model = 512
nhead = 8
dim_feedforward = 2048
dropout = 0.1
num_layer = 6
batch_size = 4
midi_encoded_path = ""
sequence_length = 1024

custom_encoder = MusicTransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                              dim_feedforward=dim_feedforward, 
                                              dropout=dropout, activation="relu")

custom_decoder = MusicTransformerDecoderLayer(d_model=d_model, nhead=nhead, 
                                              dim_feedforward=dim_feedforward, 
                                              dropout=dropout, activation="relu")

model = torch.nn.modules.Transformer(d_model=d_model, nhead=nhead, 
                                     num_encoder_layers=num_layer, 
                                     num_decoder_layers=num_layer, 
                                     dim_feedforward=dim_feedforward, 
                                     dropout=dropout, activation="relu", 
                                     custom_encoder=customer_encoder, 
                                     custom_decoder=customer_decoder)


# Optimizer
# Adam optimizer [20] with β 1 = 0.9, β 2 = 0.98 and  = 10 −9
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-09, weight_decay=0, amsgrad=False)

# Define a scheduler to vary the learning rate

lr_lambda = lambda warmup_steps

class Scheduler:
    def __init__(self, optimizer, d_model=d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.step = 0
        self.warmup_steps = warmup_steps
        
    def step(self):        
        # increment step
        self.step += 1

        # compute new learning rate        
        l_rate = self.d_model**(-.5) * min(self.step_num**(-.5), self.step_num * self.warmup_steps**(-1.5))

        # update optimizer learning rate
        for p in optimizer.param_groups['param']:
            p['lr'] = l_rate

        # update the weights in the network
        self.optimizer.step()


# See if it is possible to do it using lr_scheduler.LambdaLR lr_scheduler.StepLR
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)


# Training

for e in range(max_epochs):
    # train phase
    model.train()

    # feed data into the network and get outputs.
    inputs, labels = generate_dataset(batch_size, midi_encoded_path, sequence_length, mode)
 
    # Flush out gradients computed at the previous step before computing gradients at the current step. 
    #       Otherwise, gradients would accumulate.
    optimizer.zero_grad()
    
    # feed data into the network and get outputs.
    logits = model(inputs)
    
    # calculate loss
    loss = F.cross_entropy(logits, labels)

    # accumulates the gradient and backprogate loss.
    loss.backward()

    # performs a parameter update based on the current gradient
    scheduler.step()    
    
    # compute accuracy
    # save weights and error