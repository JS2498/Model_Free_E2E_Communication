#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 08:01:16 2022

@author: jayanths
"""

"""
Implementation  of "Model-Free Training of E2E Communication Systems"

We try to model the entire Tx-Rx system as a autoencoder which does not require
the channel model to predict the transmitted symbol at the receiver.

Transmitter  : Encoder
Receiver : Decoder

Brief overview of the implementation

Model : Autoencoder
I/p : One-hot encoded vector 'msg'
Encoder o/p : Symbol 'x' obtained from the modulation scheme corresponding to 'msg'
Decoder i/p : Noisy version of the 'x'
Decoder o/p : Predicted Output, m_estimate preceded by a soft-max layer

Encoder : 1st layer: o/p size is same as the size of 'msg', activation : ELU
          2nd layer: o/p size is of the size '2*N' where N is the no. of different
                     symbols used in the modulation scheme, activation : ReLU
          3rd layer: Batch Normalization (To satisty the power characteristics)
          

Decoder : 

"""

# import the required libraries


import numpy as np
import matplotlib.pyplot as plt
# from tensorflow.keras.layers import *

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# %%

# Helper Functions

#------------------------------AWGN Channel------------------------------------
def AWGN(X, noise_std):
    # Standard deviation is multipleid by (1/sqrt(2)) so that the standard deviation of the complex noise is 1. 
    # noise = K.random_normal(K.shape(X), mean=0, stddev=(noise_std*0.707))
    X = torch.tensor(X)
    # print(X.size())
    noise = torch.normal(0, noise_std * 0.707, X.size())
    # separate the real and imaginary parts of the noise and the symbol
    noise_r = noise[:, :, 0]
    noise_i = noise[:, :, 1]

    X_r = X[:, :, 0]
    X_i = X[:, :, 1]

    # add the real and imaginary parts of noise and the symbols

    real = X_r + noise_r
    imag = X_i + noise_i
    # stack the real and imaginary parts along the last axis and return tensor
    Y = torch.stack([real, imag], axis=-1)
    return Y


#------------------------------Perturbation-----------------------------------

# %%


def perturbation(X):

    sigma = 0.15 * 0.707  # std deviation used in the paper

    # Standard deviation is multipleid by (1/sqrt(2)) so that the
    # standard deviation of the complex perturbation is 1.
    noise = torch.normal(0, sigma, X.size())
    noise_r = noise[:, :, 0]
    noise_i = noise[:, :, 1]

    X_r = np.sqrt(1 - sigma**2) * X[:, :, 0]  # refer to eq. 11 in the paper
    X_i = np.sqrt(1 - sigma**2) * X[:, :, 1]
    real = X_r + noise_r
    imag = X_i + noise_i

    Y = torch.stack([real, imag], axis=2)
    return Y

#--------------Rayleigh Fading Channel (Rayleigh Block Fading)-----------------


def RBF(X):

    # Channel Fading coefficient is multipleid by (1/sqrt(2)) so that the
    # standard deviation of the complex channel fading coefficient is 1.
    h = 0.707 * torch.normal(0, 1, X.size())
    noise = torch.normal(0, noise_std*0.707, X.size())
    noise_r = noise[:, :, 0]
    noise_i = noise[:, :, 1]
    h_r = h[:, :, 0]
    h_i = h[:, :, 1]

    X_r = h_r * X[:, :, 0]  # refer to eq. 11 in the paper
    X_i = h_i * X[:, :, 1]

    real = X_r + noise_r
    imag = X_i + noise_i

    Y = torch.stack([real, imag], axis=2)
    return Y


def generate_one_hot_vector(M, data_size, get_label=False):
    """
    Generate one hot vectors for training and testing
    M : No. of bits in a message
    data_size : no. of messages to be generated
    """
    M_Oh = 2 ** M  # dimension of the one-hot message

    # generate random intergers between 0 to M-1
    eye_matrix = np.eye(M_Oh)  # identity matrix of size 2**M
    msgs = np.tile(eye_matrix, (int(data_size/M_Oh), 1))  # repeat the generated one-hot vectors
    np.random.shuffle(msgs)  # shuffle the messages along the 1st dimension

    return msgs


def BER(y_true, y_pred):
    # Finds the bit error rate
    return torch.mean(torch.ne(y_true, torch.round(y_pred)), axis=-1)


def B_Ber(input_msg, msg):
    '''Calculate the Block Error Rate'''
    pred_error = 1.0 * (torch.argmax(msg, 1), torch.argmax(input_msg, 1))
    # print(pred_error)
    # bber = torch.reduce_mean(tf.cast(pred_error, tf.float32))
    bber = torch.reduce_mean(pred_error)
    return bber


def AvgEngy_Constraint(X):
  X_pseudonorm = torch.sqrt(torch.reduce_mean( torch.square(X), axis=[1, 2],keepdims=True)) * 2

  X_normalized = X/X_pseudonorm
  # print(tf.reduce_sum(tf.square(X_normalized), axis = [1,2])) #for cross-checking
  return X_normalized

# %%

# Model Free E2E CS
# ----------------Parameters for the autoencoder-------------------------------
M = 8  # no. of bits in a message
N = 4  # no. of symbols corresponding to the modulation scheme

SNRdb_train = 10  # SNR in dB for which the model is trained
SNRWt_train = 10 ** (SNRdb_train/10)  # SNR in watt
noise_std = np.sqrt(1/(2*SNRWt_train))  # std deviation of noise signal

batch_size = 20 * (2**M)  # batch size
n_epoch = 15  # number of epochs
lr = 0.01  # learning rate
train_size = 2500*(2**M)  # no. of training examples
test_size = int(train_size*0.1)  # no. of test examples

# train_ds = MNIST(root = '/home/jayanths/Codes/datasets', train=True, transform=transforms.ToTensor())
# train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_norm = torch.sqrt(torch.mean(torch.square(x)) * 2)

        return x/x_norm


class AWGN(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.noise_std = args

    def forward(self, X):
        
        # print(X.size())

        noise = torch.normal(0, np.sqrt(noise_std/2), X.size())
        
        return X + noise

# %%
class Trad_AE(nn.Module):
    
    def __init__(self, input_dim, latent_dim, noise_std):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.noise_std = noise_std
    
        self.AWGN = AWGN(noise_std)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ELU(),

            nn.Linear(input_dim, 2*latent_dim),

            Reshape(-1, latent_dim, 2),
            Normalize(),
            )

        self.decoder = nn.Sequential(
            nn.Flatten(),

            nn.Linear(2 * self.latent_dim, self.input_dim),
            nn.ReLU(),

            nn.Linear(self.input_dim, self.input_dim),
            nn.Softmax()
            )

    def forward(self, x):
        
        x = x.float()
        # print(x.shape)
        z = self.encoder(x)
        # self.AWGN = AWGN()
        y = self.AWGN(z)
        x_est = self.decoder(y)

        return x_est


# %%

#------------------Genertate the dataset for training----------------------------------------
train_messages = generate_one_hot_vector(M, train_size)
# train_data = tf.data.Dataset.from_tensor_slices((train_messages,train_messages))
# train_data = train_data.shuffle(buffer_size=1024).batch(b_size)

# train_ds = MNIST(root = '/home/jayanths/Codes/datasets', train=True, transform=transforms.ToTensor())
train_dl = DataLoader(train_messages, batch_size=batch_size, shuffle=True)

# %%
#train the autoencoder model
# history = LossHistory()
# AE_NoAL.fit(train_messages,train_messages,batch_size=b_size,epochs=n_epoch, verbose=1, callbacks=[history])

loss_fn = torch.nn.CrossEntropyLoss()

def fit(model, train_loader, opt, epochs=100):
    """
    Input:
        model : Variational auto-encoder model
        train_loader : training data in batches
        opt : optimizer to reduce the loss
        epochs : no. of epochs to train the model
    """

    history = {}
    history["loss"] = []

    for epoch in range(epochs):
        model.train()  # tell the model that you are training
        batch_idx = 0
        for batch in train_loader:

            msg = batch
            msg_val = torch.argmax(msg, dim=1)
            

            # to make the gradient not to accumulate over the batches
            opt.zero_grad()

            pred_msg = model(msg)
            
            # print(pred_msg.size(), msg_val.size())

            loss = loss_fn(pred_msg, msg_val)
            loss.backward()
            opt.step()

            history['loss'].append(loss)
            if batch_idx %10 == 0:
                print(f"Epoch : {epoch}/{epochs} | batch : {batch_idx}/{len(train_loader)} | Loss : {loss}")

            batch_idx += 1

    return history

lr = 0.1
model = Trad_AE(2**M, N, noise_std)
optimizer = torch.optim.Adam(model.parameters(), betas= (0.5, 0.999), lr= lr)
history = fit(model, train_dl, optimizer, epochs=1000)

# %%

test_size = 1e6
#--------------------Plot the BLER v/s SNR plot--------------------------------
SNR_range = list(np.linspace(-2.2, 12.8, 20))
# BER = [None] * len(SNR_range)
BLER = [None] * len(SNR_range)

print("The variation of BLER with respect to SNR:")
for n in range(0, len(SNR_range)):
    SNR_Wt =10 ** (SNR_range[n]/10)
    noise_std = np.sqrt(1/(2*SNR_Wt))  #std deviation of noise signal
    
    test_messages = generate_one_hot_vector(M, test_size)  #generate the test dataset
    no_errors = 0
    X_hat = Trad_AE.encoder(test_messages)  #obtain the encoded signal
    Y = AWGN(X_hat, noise_std)   #transmit the encoded signal through the channel
    msg_hat = Trad_AE.decoder(Y)  #decode the received signal
    
    # msg_hat = AE.predict(test_messages) #this doesn't work (may be because for different SNR's AWGN noise is not changed) 
    msg_hat = msg_hat.argmax(axis = -1)
    
    msg_true = test_messages.argmax(axis = -1)
    no_errors = (msg_hat != msg_true)
    no_errors = np.sum(no_errors)
    
    # m_hat_oh = tf.one_hot(m_hat_oh,depth = 2**M)
    # m_hat_oh = m_hat_oh.numpy()
    # no_errors = (m_hat_oh != test_messages)
    # # print(no_errors[1:3,:])
    # no_errors = np.sum(no_errors,axis=1)
    # no_errors[no_errors>0] = 1
    # # print(np.max(no_errors), 'max')
    # no_errors = no_errors.astype(int).sum() #check this step tomorrow
    # # print(no_errors)
    BLER[n] = no_errors/test_size
    print(" SNR : {:.2f}, BLER : {:.6f}".format(SNR_range[n],BLER[n]))
    # print('SNR:', SNR_range[n], 'BLER', BLER[n])
