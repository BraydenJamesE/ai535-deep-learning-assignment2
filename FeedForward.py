"""
  Brayden Edwards
"""

from turtle import width
import numpy as np
import time
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
  def forward(self, logits, labels):
    self.batch_size = labels.shape[0]
    self.logits = logits
    self.labels = labels
    
    self.sigmoid_output = 1 / (1 + np.exp(-logits))
    
    loss = -np.mean(labels * np.log(self.sigmoid_output + 1e-9) + (1 - labels) * np.log(1 - self.sigmoid_output + 1e-9))
    return loss

  def backward(self):
    return (self.sigmoid_output - self.labels) / self.batch_size
    
  

class ReLU:
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
  def __init__(self, input_dim, output_dim):
    self.W = np.random.normal(0, np.sqrt(2 / input_dim), (input_dim, output_dim)) # using He intialization for ReLU
    self.b = np.zeros((1, output_dim))
    self.velocity_W = 0
    self.velocity_b = 0
    
  def forward(self, input):
    self.input = input
    return input @ self.W + self.b

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
    # handling W grads
    weight_decay_update_W = 2 * step_size * weight_decay * self.W
    new_grad_W = step_size * self.grad_weights
    
    self.velocity_W = momentum * self.velocity_W - new_grad_W - weight_decay_update_W 
    self.W = self.W + self.velocity_W
    
    # handling bias grads
    new_grad_b = step_size * self.grad_bias
    
    self.velocity_b = momentum * self.velocity_b - new_grad_b 
    self.b = self.b + self.velocity_b
    
    
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


def plot_accuracy(val_acc):
  plt.figure(figsize=(10,6))
  plt.plot(val_acc, color='red')
  plt.xlabel("Epochs")
  plt.ylabel("Validation Accuracy")
  plt.title("Final Model Validation Accuracy")
  plt.show()
  
  
def plot_accuracy_ylim(val_acc):
  plt.figure(figsize=(10,6))
  plt.plot(val_acc, color='red')
  plt.xlabel("Epochs")
  plt.ylabel("Validation Accuracy")
  plt.ylim((0,1))
  plt.title("Final Model Validation Accuracy")
  plt.show()
  

def plot_val_loss_ylim(val_loss):
  plt.figure(figsize=(10,6))
  plt.plot(val_loss, color='blue')
  plt.xlabel("Epochs")
  plt.ylabel("Validation Loss")
  plt.ylim((0,1))
  plt.title("Final Model Validation Loss")
  plt.show()


def plot_val_loss(val_loss):
  plt.figure(figsize=(10,6))
  plt.plot(val_loss, color='blue')
  plt.xlabel("Epochs")
  plt.ylabel("Validation Loss")
  plt.title("Final Model Validation Loss")
  plt.show()
  
  
def plot_accuracy_per_batch_size(accuracies_dict, batch_sizes):
  plt.figure(figsize=(10,6))
  colors = ['red', 'yellow', 'blue', 'green', 'purple', 'pink', 'brown', 'orange']

  for i in range(len(batch_sizes)):
    batch_size = batch_sizes[i]
    plt.plot(accuracies_dict[batch_size], label=batch_size, color=colors[i])

  plt.xlabel("Iterations")
  plt.ylabel("Validation Accuracy")
  plt.title("Validation Accuracy With Various Batch Sizes")

  plt.legend()
  plt.show()
  
  
def plot_accuracy_per_arg2_ylim(accuracies_dict, arg2, title, time_needed = None):
  plt.figure(figsize=(10,6))
  colors = ['red', 'orange', 'blue', 'green', 'purple', 'pink', 'brown', 'yellow']
  
  for i in range(len(arg2)):
    arg2_arg = arg2[i]
    if time_needed:
      label = str(arg2_arg) + f" ({time_needed[arg2_arg]:.2f} sec.)"
    else: 
      label = str(arg2_arg)
    plt.plot(accuracies_dict[arg2_arg], label=label, color=colors[i])

  plt.xlabel("Iterations")
  plt.ylabel("Validation Accuracy")
  plt.title(title)
  
  
  plt.ylim((0, 1))
  plt.legend()
  plt.show()
  
  
def plot_accuracy_per_arg2(accuracies_dict, arg2, title, time_needed = None):
  plt.figure(figsize=(10,6))
  colors = ['red', 'orange', 'blue', 'green', 'purple', 'pink', 'brown', 'yellow']
  
  for i in range(len(arg2)):
    arg2_arg = arg2[i]
    if time_needed:
      label = str(arg2_arg) + f" ({time_needed[arg2_arg]:.2f} sec.)"
    else: 
      label = str(arg2_arg)
      
    plt.plot(accuracies_dict[arg2_arg], label=label, color=colors[i])

  plt.xlabel("Iterations")
  plt.ylabel("Validation Accuracy")
  plt.title(title)
  
  plt.legend()
  plt.show()
  

def main():
  # hyperparameters
  max_epochs = 100
  step_size = 0.001
  number_of_layers = 2
  width_of_layers = 32
  weight_decay = 0.001
  momentum = 0.8
  batch_size = 32
 

  # Load data
  data = pickle.load(open('cifar_2class_py3.p', 'rb'))
  X_train = normalize_data(data['train_data'].astype(np.float32))
  X_test = normalize_data(data['test_data'].astype(np.float32))
  Y_train = data['train_labels']
  Y_test = data['test_labels']
  
  # Some helpful dimensions
  num_examples, input_dim = X_train.shape
  output_dim = 1 # number of class labels -1 for sigmoid loss


  # Some lists for book-keeping for plotting later
  losses = []
  val_losses = []
  accs = []
  val_accs = []
  
  loss_fn = SigmoidCrossEntropy()
  net = FeedForwardNeuralNetwork(input_dim, output_dim, width_of_layers, number_of_layers)
  # Q2 TODO: (DONE) For each epoch below max epochs
  start_time = time.time()
  for epoch in range(max_epochs): 
    
    # Scramble order of examples
    indices = np.random.permutation(num_examples)
    X_train = X_train[indices]
    Y_train = Y_train[indices]
    epoch_loss = 0
    correct_preds = 0

    # for each batch in data:
    for i in range(0, num_examples, batch_size): # for i in range(0, num_examples - (num_examples % batch_size), batch_size): throwing out the last batch if it's not full

      X_batch = X_train[i:i+batch_size]
      Y_batch = Y_train[i:i+batch_size]

      logits = net.forward(X_batch)
      if np.isnan(logits).any(): print("Warning: logits is NaN! in training")
      
      loss = loss_fn.forward(logits, Y_batch)

      epoch_loss += loss

      grad = loss_fn.backward()
      net.backward(grad)
      
      net.step(step_size, momentum, weight_decay)
      
      probabilities = 1 / (1 + np.exp(-logits))
      predictions = (probabilities > 0.5).astype(int)
      correct_preds += np.sum(predictions == Y_batch)
  
    epoch_avg_loss = epoch_loss / (num_examples // batch_size)
    epoch_avg_acc = correct_preds / num_examples
    
    val_loss, tacc = evaluate(net, X_test, Y_test, batch_size) 

    losses.append(epoch_avg_loss)
    accs.append(epoch_avg_acc)
    val_losses.append(val_loss)
    val_accs.append(tacc)
  
    logging.info("[Epoch {:3}]   Loss:  {:8.4}     Train Acc:  {:8.4}%      Val Acc:  {:8.4}%".format(epoch ,epoch_avg_loss, epoch_avg_acc * 100, tacc*100))
    
  end_time =  time.time()
  time_needed = end_time - start_time
  print(f"----- Time to train: {time_needed:.2f} --------")



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