"""
  Brayden Edwards
"""

from turtle import width
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
font = {'weight' : 'normal','size'   : 22}
matplotlib.rc('font', **font)
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')



######################################################
# Q1 Implement Init, Forward, and Backward For Layers
######################################################
class SigmoidCrossEntropy:
  # Compute the cross entropy loss after sigmoid. The reason they are put into the same layer is because the gradient has a simpler form
  # logits -- batch_size x num_classes set of scores, logits[i,j] is score of class j for batch element i
  # labels -- batch_size x 1 vector of integer label id (0,1) where labels[i] is the label for batch element i
  #
  # TODO: (DONE) Output should be a positive scalar value equal to the average cross entropy loss after sigmoid
  def forward(self, logits, labels):
    self.batch_size = labels.shape[0]
    self.logits = logits
    self.labels = labels
    
    self.sigmoid_output = 1 / (1 + np.exp(-logits))
    
    loss = -np.mean(labels * np.log(self.sigmoid_output + 1e-9) + (1 - labels) * np.log(1 - self.sigmoid_output + 1e-9))
    return loss

  # TODO: (DONE) Compute the gradient of the cross entropy loss with respect to the the input logits 
  def backward(self):
    return (self.sigmoid_output - self.labels) / self.batch_size
    
  

class ReLU:
  # TODO: (DONE)Compute ReLU(input) element-wise
  def forward(self, input):
    self.input = input # storing the original input for backprop
    return np.maximum(0, input)
      
  # TODO: (DONE) Given dL/doutput, return dL/dinput
  def backward(self, grad):
    return grad * (self.input > 0)
  
  # No parameters so nothing to do during a gradient descent step
  def step(self,step_size, momentum = 0, weight_decay = 0):
    return


class LinearLayer:
  # TODO: (DONE) Initialize our layer with (input_dim, output_dim) weight matrix and a (1,output_dim) bias vector
  def __init__(self, input_dim, output_dim):
    self.W = np.random.uniform(-0.05, 0.05, (input_dim, output_dim))
 # np.random.rand(input_dim, output_dim)
    self.b = np.zeros((1, output_dim))
    self.velocity_W = 0
    self.velocity_b = 0
    
  # TODO: (DONE) During the forward pass, we simply compute XW+b
  def forward(self, input):
    self.input = input
    return input @ self.W + self.b

  # TODO: (DONE) Backward pass inputs:
  #
  # grad dL/dZ -- For a batch size of n, grad is a (n x output_dim) matrix where 
  #         the i'th row is the gradient of the loss of example i with respect 
  #         to z_i (the output of this layer for example i)

  # Computes and stores:
  # 
  # self.grad_weights dL/dW --  A (input_dim x output_dim) matrix storing the gradient
  #                       of the loss with respect to the weights of this layer. 
  #                       This is an summation over the gradient of the loss of
  #                       each example with respect to the weights.
  #
  # self.grad_bias dL/dZ--     A (1 x output_dim) matrix storing the gradient
  #                       of the loss with respect to the bias of this layer. 
  #                       This is an summation over the gradient of the loss of
  #                       each example with respect to the bias.
  
  # Return Value:
  #
  # grad_input dL/dX -- For a batch size of n, grad_input is a (n x input_dim) matrix where
  #               the i'th row is the gradient of the loss of example i with respect 
  #               to x_i (the input of this layer for example i) 

  def backward(self, grad):
    self.grad_weights = self.input.T @ grad 
    self.grad_bias = np.sum(grad, axis=0, keepdims=True)
    self.grad_input = grad @ self.W.T
    return self.grad_input
    

  ######################################################
  # Q2 Implement SGD with Weight Decay
  ######################################################  
  def step(self, step_size, momentum = 0.8, weight_decay = 0.0):
  # TODO: Implement the step
    # handling W grads
    weight_decay_update_W = 2 * step_size * weight_decay * self.W
    new_grad_W = step_size * self.grad_weights
    
    self.velocity_W = momentum * self.velocity_W - new_grad_W - weight_decay_update_W 
    self.W = self.W + self.velocity_W
    
    # handling bias grads
    #weight_decay_update_b = 2 * step_size * weight_decay * self.b # Found an article telling me to not use weight decay on the bias term. May consider assessing the performance with and without. 
    new_grad_b = step_size * self.grad_bias
    
    self.velocity_b = momentum * self.velocity_b - new_grad_b #- weight_decay_update_b # see comment above for including weight decay or not
    self.b = self.b + self.velocity_b
    


######################################################
# Q4 Implement Evaluation for Monitoring Training
###################################################### 
# TODO: Given a model, X/Y dataset, and batch size, return the average cross-entropy loss and accuracy over the set
def evaluate(model, X_val, Y_val, batch_size):
  loss_fn = SigmoidCrossEntropy()
  
  num_examples = X_val.shape[0]
  total_loss = 0
  correct_preds = 0
  
  for i in range(0, num_examples, batch_size):
    X_batch = X_val[i:i+batch_size]
    Y_batch = Y_val[i:i+batch_size]
    logits = model.forward(X_batch)
    if np.isnan(logits).any(): print("Warning: logits is NaN! in testing")
    loss = loss_fn.forward(logits, Y_batch)
    total_loss += loss
    
    probabilities = 1 / (1 + np.exp(-logits))
    predictions = (probabilities > 0.5).astype(int)
    correct_preds += np.sum(predictions == Y_batch)
    
  avg_loss = total_loss / (num_examples // batch_size)
  accuracy = correct_preds / num_examples
  
  return avg_loss, accuracy

def normalize_data(data):
  std = np.std(data)
  mean = np.mean(data)
  return (data - mean) / std

def plot_train_val_accuracy(train_info, val_info):
    fig, ax = plt.subplots(figsize=(16,9))  # Correct way to get an axis
    ax.set_ylim(0, 1)

    ax.plot(train_info, c="blue", label="Train Accuracy")
    ax.plot(val_info, c="red", label="Validation Accuracy")

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Accuracy")
    ax.legend()  
    plt.show()
  
def plot_train_loss(train_loss):
    fig, ax = plt.subplots(figsize=(16,9))  # Correct way to get an axis
    ax.set_ylim(0, 1)

    ax.plot(train_loss, c="blue", label="Train Loss")

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Avg. Cross-Entropy Loss")
    ax.legend()  
    plt.show()

def main():
  # TODO: Set optimization parameters (NEED TO SUPPLY THESE)
  batch_size = 32
  max_epochs = 50
  step_size = 0.0001

  number_of_layers = 3
  width_of_layers = 64
  weight_decay = 0.0001
  momentum = 0.5

  # Load data
  data = pickle.load(open('cifar_2class_py3.p', 'rb'))
  X_train = normalize_data(data['train_data'].astype(np.float32))
  X_test = normalize_data(data['test_data'].astype(np.float32))
  Y_train = data['train_labels']
  Y_test = data['test_labels']
  
  # Some helpful dimensions
  num_examples, input_dim = X_train.shape
  output_dim = 1 # number of class labels -1 for sigmoid loss


  # Build a network with input feature dimensions, output feature dimension,
  # hidden dimension, and number of layers as specified below. You can edit this as you please.
  net = FeedForwardNeuralNetwork(input_dim, output_dim, width_of_layers, number_of_layers)

  # Some lists for book-keeping for plotting later
  losses = []
  val_losses = []
  accs = []
  val_accs = []
  
  loss_fn = SigmoidCrossEntropy()
  
  # Q2 TODO: For each epoch below max epochs
  for epoch in range(max_epochs): 
    
    # Scramble order of examples
    indices = np.random.permutation(num_examples)
    X_train = X_train[indices]
    Y_train = Y_train[indices]
    epoch_loss = 0
    correct_preds = 0

    # for each batch in data:
    for i in range(0, num_examples - (num_examples % batch_size), batch_size): # throwing out the last batch if it's not full
      # Gather batch
      X_batch = X_train[i:i+batch_size]
      Y_batch = Y_train[i:i+batch_size]

      # Compute forward pass
      logits = net.forward(X_batch)
      if np.isnan(logits).any(): print("Warning: logits is NaN! in training")
      
      # Compute loss
      loss = loss_fn.forward(logits, Y_batch)
      
      epoch_loss += loss

      # Backward loss and networks
      grad = loss_fn.backward()
      net.backward(grad)
      
      # Take optimizer step
      net.step(step_size, momentum, weight_decay)
      
      # Book-keeping for loss / accuracy
      probabilities = 1 / (1 + np.exp(-logits))
      predictions = (probabilities > 0.5).astype(int)
      correct_preds += np.sum(predictions == Y_batch)
  
    # Evaluate performance on test.
    epoch_avg_loss = epoch_loss / (num_examples // batch_size)
    epoch_avg_acc = correct_preds / num_examples
    
    val_loss, tacc = evaluate(net, X_test, Y_test, batch_size) 

    losses.append(epoch_avg_loss)
    accs.append(epoch_avg_acc)
    val_losses.append(val_loss)
    val_accs.append(tacc)
    
    
    ###############################################################
    # Print some stats about the optimization process after each epoch
    ###############################################################
    # epoch_avg_loss -- average training loss across batches this epoch
    # epoch_avg_acc -- average accuracy across batches this epoch
    # vacc -- testing accuracy this epoch
    ###############################################################
    
    logging.info("[Epoch {:3}]   Loss:  {:8.4}     Train Acc:  {:8.4}%      Val Acc:  {:8.4}%".format(epoch ,epoch_avg_loss, epoch_avg_acc * 100, tacc*100))
  
  # plot_train_val_accuracy(train_info=accs, val_info=val_accs)
  # plot_train_loss(train_info=losses)
    
  ###############################################################
  # Code for producing output plot requires
  ###############################################################
  # losses -- a list of average loss per batch in training
  # accs -- a list of accuracies per batch in training
  # val_losses -- a list of average testing loss at each epoch
  # val_acc -- a list of testing accuracy at each epoch
  # batch_size -- the batch size
  ################################################################
  
  
  fig = plt.figure(figsize=(16, 9))

  # ---- FIRST PLOT: Training & Validation Loss ----
  plt.clf()  # Clear the figure to prepare for the first plot
  plt.plot(range(len(losses)), losses, c="blue", label="Train Loss")
  plt.plot(range(len(val_losses)), val_losses, c="red", label="Validation Loss")

  plt.xlabel("Epochs")
  plt.ylabel("Avg. Cross-Entropy Loss")
  plt.legend(loc="upper right")
  plt.ylim(0, max(max(losses), max(val_losses)) * 1.2)  # Dynamically scale Y-axis
  plt.title("Training vs Validation Loss")

  plt.draw()  # Draw the figure but don't close it
  plt.pause(1)  # Pause for interaction

  # ---- SECOND PLOT: Training & Validation Accuracy ----
  plt.clf()  # Clear figure to prepare for the second plot
  plt.plot(range(len(accs)), accs, c="blue", label="Train Accuracy")
  plt.plot(range(len(val_accs)), val_accs, c="red", label="Validation Accuracy")

  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.legend(loc="lower right")
  plt.ylim(0, 1.01)  # Ensure accuracy stays between 0-1
  plt.title("Training vs Validation Accuracy")

  plt.draw()  # Update the figure
  plt.pause(1)  # Pause for interaction

  # Keep the window open and interactive
  plt.show()

#####################################################
# Feedforward Neural Network Structure
# -- Feel free to edit when tuning
#####################################################

class FeedForwardNeuralNetwork:

  def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
    self.layers = []
    if num_layers == 1:
      self.layers.append(LinearLayer(input_dim, output_dim))
    else:
    # TODO: Please create a network with hidden layers based on the parameters
      self.layers.append(LinearLayer(input_dim, hidden_dim))
      self.layers.append(ReLU())
      for _ in range(num_layers - 2):
        self.layers.append(LinearLayer(hidden_dim, hidden_dim))
        self.layers.append(ReLU())
      self.layers.append(LinearLayer(hidden_dim, output_dim))
        
  
  def forward(self, X):
    for layer in self.layers:
      X = layer.forward(X)
    return X

  def backward(self, grad):
    for layer in reversed(self.layers):
      grad = layer.backward(grad)

  def step(self, step_size, momentum, weight_decay):
    for layer in self.layers:
      layer.step(step_size, momentum, weight_decay)



def displayExample(x):
  r = x[:1024].reshape(32,32)
  g = x[1024:2048].reshape(32,32)
  b = x[2048:].reshape(32,32)
  
  plt.imshow(np.stack([r,g,b],axis=2))
  plt.axis('off')
  plt.show()


if __name__=="__main__":
  main()