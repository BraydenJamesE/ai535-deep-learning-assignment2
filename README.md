# ai535-deep-learning-assignment2

## Neural Network Hyperparameter Tuning

Author: Brayden Edwards
Date: February 9, 2025

## Overview

This project implements a neural network from scratch, without using TensorFlow, PyTorch, or other deep learning frameworks. The NN was trained on a 2-class subset of CIFAR-10 using ReLU activation and binary cross-entropy loss. The goal is to tune hyperparameters to maximize validation accuracy while preventing overfitting. A final report can be found in the files on Github. 

## Dataset

Training: 10,000 examples (3072 features)
Testing: 2,000 examples
Task: Binary classification
Hyperparameter Tuning

The model was tested with different batch sizes, learning rates, hidden layers, weight decay, and momentum to find the best configuration.

## Final Model

max_epochs = 100
step_size = 0.001
number_of_layers = 2
width_of_layers = 32
weight_decay = 0.001
momentum = 0.8
batch_size = 32

## Results

Final Model Accuracy: 84.4%
