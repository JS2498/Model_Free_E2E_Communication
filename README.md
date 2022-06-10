# Model Free Training of End-to-End Communication Systems
Implementation of the paper **"Model Free Training of End-to-End Communication Systems"** - Fayçal Ait Aoudia, Jakob Hoydis. 

The authors in the paper "Model-Free Training of End-to-End Communication Systems" consider *auto-encoder* based architecture for the entire transmitter and receiver model with the channel between the encoder (transmitter) and decoder (receiver). The following is the brief explaination of the paper.

* Here, *model-free* refers to the communication model where the channel is unknown or has non-differentiable components. Hence the encoder (transmitter) cann't be trained through backpropagation.

* The objective was to train the autoencoder such that the decoder is able to detect the transmitted messages that are transmitted through the channel.

* To train the autoencoder, the authors propose an alternating algorithm where the decoder (receiver) and the encoder (transmitter) are trained separately.
  
* Since the encoder cann't be trained as the channel can be non-differentiable, approximate gradient of loss function is used to train the encoder (transmitter).

#### *Neural Network Architecture:*

* **Input**: One-hot encoded messages that are transmitted through AWGN/ Rayleigh Block Fading (RBF) channel after encoding.
  
* **Encoder**: 2 layers of fully connected network with *elu* as the activation function with normalization to satisfy the power constraints. The output of the encoder is the symbol that is transmitted.

* **Decoder**: 2 layers of fully connected network with *elu* as the activation function for the $1^{st}$ layer and *softmax* as the activation function for the $2^{nd}$ layer. Input to the decoder is the noisy version of the transmitted symbol.
  
* **Output**: Predict the transmitted symbol using the received symbol.

* **Loss Function**: Categorical cross-entropy
  
* **Optimizer**: Adam

#### References
* [Model Free Training of End-to-End Communication Systems](https://ieeexplore.ieee.org/document/8792076)
* [Aithu-Snehith's implementation of End to End learning of Communication Systems without a channel model](https://github.com/Aithu-Snehith/End-to-End-Learning-of-Communications-Systems-Without-a-Channel-Model)
