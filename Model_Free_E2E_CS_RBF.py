# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 19:48:10 2022

@author: Jayanth S
"""

# import the required libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
# from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

# %%

# ------------------------------AWGN Channel----------------------------------


def AWGN(X, noise_std):
    # Standard deviation is multipleid by (1/sqrt(2)) so that the standard
    # deviation of the complex noise is 1.
    noise = K.random_normal(K.shape(X), mean=0, stddev=(noise_std*0.707))
    Y = tf.keras.layers.Add()([X, noise])
    # separate the real and imaginary parts of the noise and the symbol
    # noise_r = noise[:,:,0]
    # noise_i = noise[:,:,1]

#     X_r = X[:,:,0]
#     X_i = X[:,:,1]

#     #add the real and imaginary parts of noise and the symbols
#     real = tf.keras.layers.Add()([X_r, noise_r])
#     imag = tf.keras.layers.Add()([X_i, noise_i])

#     #stack the real and imaginary parts along last axis and return the tensor
#     Y = K.stack([real, imag], axis=-1)
    return Y

# ------------------------------Perturbation------------------------------------


def perturbation(X):
    sigma = 0.15*0.707  # std deviation used in the paper

    # Standard deviation is multipleid by (1/sqrt(2)) so that the standard
    # deviation of the complex perturbation is 1.
    noise = K.random_normal(K.shape(X), mean=0, stddev=sigma)
    noise_r = noise[:, :, 0]
    noise_i = noise[:, :, 1]

    X_r = tf.sqrt(1-sigma**2) * X[:, :, 0]  # refer to eq. 11 in the paper
    X_i = tf.sqrt(1-sigma**2) * X[:, :, 1]
    real = tf.keras.layers.Add()([X_r, noise_r])
    imag = tf.keras.layers.Add()([X_i, noise_i])

    Y = K.stack([real, imag], axis=-1)
    return Y

# --------------Rayleigh Fading Channel (Rayleigh Block Fading)---------------


def RBF(X, noise_std):

    # Channel Fading coefficient is multipleid by (1/sqrt(2)) so that the
    # standard deviation of the complex channel fading coefficient is 1.
    h = K.random_normal(shape=[K.shape(X)[0], 1, 2], mean=0, stddev=0.707)
    noise = K.random_normal(K.shape(X), mean=0, stddev=noise_std*0.707)

    # hX = tf.keras.layers.Multiply()([h,X])
    # Y = tf.keras.layers.Add()([hX, noise])

    noise_r = noise[:, :, 0]
    noise_i = noise[:, :, 1]

    h_r = h[:, :, 0]
    h_i = h[:, :, 1]
    # h_r = h_r.unsqueeze(1).repeat(1, N, 1)
    X_r = h_r * X[:, :, 0]  # tf.keras.layers.Multiply()([h_r , X[:,:,0]])
    X_i = h_i * X[:, :, 1]  # tf.keras.layers.Multiply()( [h_i, X[:,:,1]])

    real = X_r + noise_r  # tf.keras.layers.Add()([X_r, noise_r])
    imag = X_i + noise_i  # tf.keras.layers.Add()([X_i, noise_i])

    Y = K.stack([real, imag], axis=-1)
    return Y


def generate_one_hot_vector(M, data_size, get_label=False):
    """
    Generate one hot vectors for training and testing
    M : No. of bits in a message
    data_size : no. of messages to be generated
    """
    M_Oh = 2**M  # dimension of the one-hot message

    # generate random intergers between 0 to M-1
    eye_matrix = np.eye(M_Oh)  # identity matrix of size 2**M
    # repeat the generated one-hot vectors
    msgs = np.tile(eye_matrix, (int(data_size/M_Oh), 1))
    np.random.shuffle(msgs)  # shuffle the messages along the 1st dimension

    return msgs


def BER(y_true, y_pred):
    # Finds the bit error rate
    return K.mean(K.not_equal(y_true, K.round(y_pred)), axis=-1)


def B_Ber(input_msg, msg):
    '''Calculate the Block Error Rate'''
    pred_error = tf.not_equal(tf.argmax(msg, 1), tf.argmax(input_msg, 1))
    # print(pred_error)
    bber = tf.reduce_mean(tf.cast(pred_error, tf.float32))
    return bber


def AvgEngy_Constraint(X):
    X_pseudonorm = tf.sqrt(tf.reduce_mean(tf.square(X), axis=[1, 2],
                                          keepdims=True)*2)

    X_normalized = X/X_pseudonorm
    # print(tf.reduce_sum(tf.square(X_normalized),axis = [1,2]))#cross-checking
    return X_normalized

# Custom callback to stor the losses for each batch


class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        # self.val_losses = []
        self.B_Ber = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        # self.val_losses.append(logs.get('val_loss'))
        self.B_Ber.append(logs.get('B_Ber'))


perturbation_layer = tf.keras.layers.Lambda(lambda x: perturbation(x))
channel_layer = tf.keras.layers.Lambda(lambda x: RBF(x, noise_std))
loss_object_tx = tf.keras.losses.CategoricalCrossentropy()
loss_object_rx = tf.keras.losses.CategoricalCrossentropy()


def decoder_loss(mb, yb,  validation=False):
    # mb : messages in batch, yb : labels in batch
    x = encoder_AL(mb)
    y = RBF(x, noise_std)
    m_est = decoder_AL(y)
#     print("decoeder", noise_std)
    # if validation == True:
    #   m_hat = tf.math.argmax(m_est, axis=1)
    #   m_hat = m_hat.numpy()
    #   # print("msg_hat : ", m_hat[0:100])
    #   m_argmax = tf.math.argmax(mb, axis=1)
    #   m_argmax = m_argmax.numpy()
    #   # print("msg : ",  m_hat_argmax[0:100])

    #   no_errors = (m_hat != m_argmax)
    #   # print(no_errors[1:3,:])
    #   no_errors = np.sum(no_errors,axis=1)
    #   # no_errors[no_errors>0] = 1
    #   # print(np.max(no_errors), 'max')
    #   # no_errors = no_errors.astype(int).sum() #check this step tomorrow
    #   print(no_errors/test_size)
#     print(m_est.shape)
    return loss_object_rx(yb, m_est), B_Ber(yb, m_est)


def encoder_loss(mb, yb):
    x = encoder_AL(mb)
    # pass the encoder output through the perturbation layer
    x_pb = perturbation_layer(x)
    y = RBF(x_pb, noise_std)
    m_est = decoder_AL(y)
#     print("encoder", noise_std)
    return loss_object_tx(yb, m_est), B_Ber(yb, m_est)

# %%


class Equalization(tf.keras.layers.Layer):
    def __init__(self):
        super(Equalization, self).__init__()
        self.h_eq = tf.Variable(tf.zeros(shape=(N, N), dtype=tf.float32),
                                dtype=tf.float32, trainable=False)
        # identity_mat = tf.Variable(tf.zeros(2,2), dtype = tf.float32)

    def build(self, input_shape):
        self.ip_shape = input_shape

    @tf.function
    def call(self, h, Y):

        # ####################################################################
        # ############## Zero Forcing Equalizer ##############################
        # ####################################################################
        new_h = tf.complex(h[:, 0], h[:, 1])
        new_h = tf.reshape(new_h, shape=[-1, 1])
        norm_val = tf.square(tf.norm(new_h, axis=-1))
        # norm_val = tf.math.real(norm_val)
        # print(new_h, new_Y)
        # print(norm_val)
        norm_val = tf.reshape(norm_val, shape=[-1, 1])
        new_h = tf.math.conj(new_h)

        new_Y = tf.complex(Y[:, :, 0], Y[:, :, 1])
        # norm_val = tf.reduce_sum(tf.square(h))

        Y_est_cmplx = tf.keras.layers.Multiply()([new_h, new_Y])
        # Y_est_cmplx = new_h * new_Y
        # Y_est_cmplx = Y_est_cmplx/norm_val
        Y_est_cmplx = tf.math.divide(Y_est_cmplx, norm_val)
        Y_est_real = tf.math.real(Y_est_cmplx)
        Y_est_imag = tf.math.imag(Y_est_cmplx)
        return tf.stack([Y_est_real, Y_est_imag], axis=-1)
        # return Y

        # ----------------------------------------------------------------------
        # y_shape  = tf.shape(Y)[0]
        # # h = hY[0]
        # h = tf.reshape(h, shape = [-1,2,1])
        # # Y = hY[1]
        # Y = tf.reshape(Y, shape = [-1,N,2])
        # # h_unstack  = tf.unstack(h)
        # # Y_unstack = tf.unstack(Y)
        # # self.y_eq = []
        # y_eq = []
        # for i in range(b_size):
        # # for hi,yi in (h_unstack,Y_unstack):
        # # print(h.shape)
        #     h_1 = h[i,0,0]
        #     h_2 = h[i,1,0]

        #     # print(h_1)
        #     identity_mat = tf.eye(2)
        #     # self.h_eq[0,0].assign(h_1)
        #     # self.h_eq[0,1].assign(h_2)
        #     # self.h_eq[1,0].assign(-h_2)
        #     # self.h_eq[1,1].assign(h_1)

        #     #the below method didn't work

        #     norm_val = tf.square(h_1) + tf.square(h_2)
        #     # self.h_eq[0:2,0:2].assign(tf.math.multiply(h_1,identity_mat))
        #     # self.h_eq[0:2,2:4].assign(tf.math.multiply(h_2,identity_mat))
        #     # self.h_eq[2:4,0:2].assign(tf.math.multiply(h_2,identity_mat))
        #     # self.h_eq[2:4,2:4].assign(tf.math.multiply(h_1,identity_mat))

        #     # self.h_eq[0:2,0:2].assign(h_1*identity_mat)
        #     # self.h_eq[0:2,2:4].assign(h_2*identity_mat)
        #     # self.h_eq[2:4,0:2].assign(-h_2*identity_mat)
        #     # self.h_eq[2:4,2:4].assign(h_1*identity_mat)

        #     Y_est = tf.linalg.matmul(self.h_eq,  Y[i])
        #     Y_est = Y_est/norm_val
        #     y_eq.append(Y_est)
        # # # self.y_eq.append(hy)
        # return tf.convert_to_tensor(y_eq, dtype = tf.float32)


# %%
# The below function for equalization didn't work out.
# Hence we built the custom layer
# h_eq = tf.Variable(K.zeros(shape = (2,2)), dtype = tf.float32)
# @tf.function
def equalization(h, Y):
    # global h_eq

    new_h = tf.complex(h[:, 0], h[:, 1])
    new_h = tf.reshape(new_h, shape=[-1, 1])
    new_Y = tf.complex(Y[:, :, 0], Y[:, :, 1])

    norm_val = tf.square(tf.norm(new_h, axis=-1))
    # norm_val = tf.math.real(norm_val)
    # print(new_h, new_Y)
    # print(norm_val)
    norm_val = tf.reshape(norm_val, shape=[-1, 1])
    # norm_val = tf.reduce_sum(tf.square(h))

    Y_est_cmplx = tf.keras.layers.Multiply()([new_h, new_Y])
    # Y_est_cmplx = new_Y*new_h
    Y_est_cmplx = tf.math.divide(Y_est_cmplx, norm_val)
    Y_est_real = tf.math.real(Y_est_cmplx)
    Y_est_imag = tf.math.imag(Y_est_cmplx)
    return tf.stack([Y_est_real, Y_est_imag], axis=-1)
    # h_2 = h[1]

    # # identity_mat = tf.eye(1)
    # print(h.shape)
    # h_eq[0,0].assign(h_1)
    # h_eq[0,1].assign(h_2)
    # h_eq[1,0].assign(-h_2)
    # h_eq[1,1].assign(h_1)

    # print(h_eq.shape, Y.shape)
    # norm_val = tf.reduce_sum(tf.square(h))
    # Y_est = tf.linalg.matmul(Y, h_eq)
    # Y_est = Y_est/norm_val

    # y_eq = tf.Variable(Y, trainable = False)
    # h_unstack  = tf.unstack(h, axis = 0)
    # Y_unstack = tf.unstack(Y, axis = 0)
    # y_eq = []
    # for i in range(500):
    #     # print(hi)
    #     h_1 = h[i,0]
    #     h_2 = h[i,1]
    #     identity_mat = tf.eye(2)
    #     h_eq[0:2,0:2].assign(h_1*identity_mat)
    #     h_eq[0:2,2:4].assign(h_2*identity_mat)
    #     h_eq[2:4,0:2].assign(-h_2*identity_mat)
    #     h_eq[2:4,2:4].assign(h_1*identity_mat)
    #     hy = tf.linalg.matmul(h_eq,Y[i])
    #     y_eq.append(hy)
    # return tf.convert_to_tensor(Y_est, dtype = tf.float32)


# %%
# ----------------Parameters for the autoencoder------------------------------
M = 8  # no. of bits in a message
N = 5  # no. of symbols corresponding to the modulation scheme

SNRdb_train = 20  # SNR in dB for which the model is trained
SNRWt_train = 10**(SNRdb_train/10)  # SNR in watt
noise_std = np.sqrt(1/(2*SNRWt_train))  # std deviation of noise signal

b_size = 128  # batch size
n_epoch = 25  # number of epochs
lr = 0.001  # learning rate
train_size = 2000*(2**M)  # no. of training examples
test_size = int(train_size*0.1)  # no. of test examples

# ------------------------Autoencoder------------------------------------

m_ip = keras.Input(shape=(2**M, ))  # input to the autoencoder

# two FC layers with 'relu' as the activation function
m = tf.keras.layers.Dense(2**M, activation='elu', name='Tx1')(m_ip)
# m = tf.keras.layers.Dense(M*M,activation = 'elu', name = 'Tx11')(m)
m = tf.keras.layers.Dense(2*N, activation=None, name='Tx2')(m)

# reshape X_hat such that it represents complex symbol
X_reshape = tf.keras.layers.Reshape((-1, 2))(m)  # Tx symbol

# we need to have a custom layer for normalization. So we use Lambda layer
# to define our own function of the custom layer.
# X = tf.keras.layers.Lambda(lambda x: K.l2_normalize(x, axis=-1))(X_reshape)

# is the below line correct? I this yes! because on average the norm by
# the number of complex symbols should be one but not each sample norm
# reduce_mean will divide the sum by 2*N (no. of terms in each encoded
# sample) hence we multiply it be 2
X = tf.keras.layers.Lambda(lambda x: x/tf.sqrt(tf.reduce_mean(
                                               tf.square(x))*2))(X_reshape)
# X = tf.keras.layers.BatchNormalization()(X)
# X = tf.keras.layers.Lambda(lambda x : AvgEngy_Constraint(x))(X_reshape)

encoder_AL = keras.Model(m_ip, X)
encoder_AL.summary()

# ########################################################################
# add the noise to the transmitted symbol
# Y = tf.keras.layers.Lambda(lambda x: RBF(x, noise_std))(X)

# ------------------------------Transofrmer Network ----------------------

X_enc = keras.Input(shape=(N, 2, ))
Y_ip = tf.keras.layers.Flatten()(X_enc)

# two FC layes at receiver with the last layer having 'softmax' activation
h_1 = tf.keras.layers.Dense(2*N, activation='tanh', name="h1")(Y_ip)
h_2 = tf.keras.layers.Dense(2, activation=None, name="h2")(h_1)

#########################################################################

# Y_reshape = tf.keras.layers.Reshape((-1,2))(Y_ip)
equalization_layer = Equalization()
trans_signal = equalization_layer(h_2, X_enc)
# trans_signal = tf.keras.layers.Lambda(lambda hY: equalization(hY[0], hY[1]))([h_2, X_enc])
# ########################################################################

trans_signal = tf.keras.layers.Flatten()(trans_signal)
Y = tf.keras.layers.Dense(2**M, activation='relu', name='Rx1')(trans_signal)
deco_op = tf.keras.layers.Dense(2**M, activation='softmax', name='Rx2')(Y)

decoder_AL = keras.Model(X_enc, deco_op)
decoder_AL.summary()
# #########################################################################

# create the autoencoder,encoder and decoder model for training and testing
AE_ip = keras.Input(shape=(2**M, ))  # input to the autoencoder
encoded_msg = encoder_AL(AE_ip)
rx_enmsg = tf.keras.layers.Lambda(lambda x: RBF(x, noise_std))(encoded_msg)
# channel_layer(encoded_msg) # received encoded message
decoded_msg = decoder_AL(rx_enmsg)

AE_AL = keras.Model(AE_ip, decoded_msg)
AE_AL.summary()
loss_fn = keras.losses.CategoricalCrossentropy()
AE_AL.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
              loss=loss_fn, metrics=[B_Ber])

train_messages = generate_one_hot_vector(M, train_size)
train_data = tf.data.Dataset.from_tensor_slices((train_messages, train_messages))
train_data = train_data.shuffle(buffer_size=256).batch(b_size)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

# AE_AL.fit(train_messages,train_messages,batch_size=b_size,
            # epochs = n_epoch, validation_split= 0.1 , verbose = 1)
# Tx and Rx training for AWGN channel
# store the accuracy history
BLER_hist_tx = [0]
BLER_hist_rx = [0]
for i in range(n_epoch):
    # j = 0
    for (x_batch, y_batch) in train_data:
        # j = j+1
        # Rx trianing
        with tf.GradientTape() as tape_rx:
            loss_value, BLER_value_rx = decoder_loss(x_batch, y_batch, validation=False)
            BLER_hist_rx.append(BLER_value_rx.numpy())
#         if (j%12 == 0): #len(list(train_data))
        grads = tape_rx.gradient(loss_value, decoder_AL.trainable_variables)
        optimizer.apply_gradients(zip(grads, decoder_AL.trainable_variables))

    # Tx training
#     for (x_batch,y_batch) in train_data:
#         j = j + 1
        with tf.GradientTape() as tape_tx:
            loss_value, BLER_value_tx = encoder_loss(x_batch, y_batch)
            BLER_hist_tx.append(BLER_value_tx.numpy())
#         if (j%12 == 0): #len(list(train_data))
        grads = tape_tx.gradient(loss_value, encoder_AL.trainable_variables)
        optimizer.apply_gradients(zip(grads, encoder_AL.trainable_variables))  

    test_messages = generate_one_hot_vector(M, test_size)
    val_loss, val_BLER = decoder_loss(test_messages, test_messages, validation=True)

    print("Epoch {:d}: Loss: {:.3f}, Training accuracy: {:.3%}, Val_loss: {:.3f}, Validation accuracy: {:.3%}".format(i, loss_value, 1-BLER_value_rx, val_loss, 1-val_BLER))

# %%

# #plot the BLER v/s Iterations
# fig = plt.figure(figsize=(5, 5))
# plt.plot(BLER_hist_rx, 'r-', label ='Receiver BLER')
# plt.plot(BLER_hist_tx, 'b-', label ='Transmitter BLER')
# plt.gca().set_ylim(0, 1)
# # plt.gca().set_xlim(0, 150)
# # plt.yscale('log')
# plt.xlabel('Iterations')
# plt.ylabel('BLER')
# plt.title('Only AWGN Channel')
# plt.grid(True, which="both")
# plt.legend()
# plt.show()

# %%

test_size = 1e5
# -------------------Plot the BLER v/s SNR plot----------------------------
SNR_range = list(np.linspace(-2, 40, 20))
# BER = [None] * len(SNR_range)
BLER = [None] * len(SNR_range)

print("Variation of BLER with respect to the SNR:")
for n in range(0, len(SNR_range)):
    SNR = 10 ** (SNR_range[n]/10)
    noise_std = np.sqrt(1/(2*SNR))  # std deviation of noise signal

    test_messages = generate_one_hot_vector(M, test_size)  # generate the test dataset
    no_errors = 0
    X_hat = encoder_AL.predict(test_messages)  # obtain the encoded signal
    Y = RBF(X_hat, noise_std)   # transmit the encoded signal through the channel
    msg_hat = decoder_AL.predict(Y)  # decode the received signal
    msg_hat = msg_hat.argmax(axis=-1)

    msg_true = test_messages.argmax(axis=-1)
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
    print(" SNR : {:.2f}, BLER : {:.6f}".format(SNR_range[n], BLER[n]))
    # print('SNR:', SNR_range[n], 'BLER', BLER[n])

# %%
# plot the BLER v/s SNR
fig = plt.figure(figsize=(8, 5))
plt.semilogy(SNR_range, BLER, 'ro-', label='Alternate Learning')
# plt.gca().set_ylim(1e-6, 1)
plt.gca().set_xlim(-4, 40)
plt.yscale('log')
plt.xlabel('SNR(dB)')
plt.ylabel('BLER')
plt.title('Only AWGN Channel')
plt.grid(True, which="both")
plt.legend()
plt.show()

BLER_AL = BLER  # store the BLER for different SNR's (useful for the future plots!)
