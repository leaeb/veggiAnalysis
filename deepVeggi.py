import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.optimizers import SGD, Adam, Adadelta, Adagrad, Nadam, RMSprop, schedules
from tensorflow.keras.utils import normalize
import tensorflow.keras.datasets as tfds
import tensorflow.keras.initializers as tfi
import tensorflow.keras.regularizers as tfr

###--------
# load data
###--------
from csv import reader
# open file in read mode
target=[]
input=[]
with open('data.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        # row variable is a list that represents a row in csv
        t,i=row
        target.append(bool(t))
        i=i.strip('][').split(', ')
        input.append(list(map(int, i)))


test_input=np.array(input[:5000])
#test_input=test_input.reshape(test_input.shape[0],len(test_input[0]))
print(test_input.shape)
test_target=np.array(target[:5000])
# Reserve samples for validation
validation_input = np.array(input[5001:10000])
validation_target = np.array(target[5001:10000])
training_input= np.array(input[10000:])

training_target=np.array(target[10000:])

print("training input shape: %s, training target shape: %s"  % (training_input.shape, training_target.shape))
print("validation input shape: %s, validation target shape: %s"  % (validation_input.shape, validation_target.shape))
print("test input shape: %s, test target shape: %s"  % (test_input.shape, test_target.shape))
# range of input values: 0 ... 255
print("\n")

###-----------
# process data
###-----------

# Note: shuffling 


num_classes = 2 # 10 digits

###-----------
# define model
###-----------
print('-----------------------------------------')
print(len(training_input[1]))
print(training_input.shape)
num_inputs = len(training_input[0])
num_hidden = [50,50] # FIX!!!
num_outputs = num_classes 

initialLearningRate = 0.02 # FIX!!!
# select constant learning rate or (flexible) learning rate schedule,
# i.e. select one of the following two alternatives
lr_schedule = initialLearningRate # constant learning rate
# lr_schedule = schedules.ExponentialDecay(initial_learning_rate = initialLearningRate, decay_steps=100000, decay_rate=0.96, staircase=True) # or PiecewiseConstantDecay or PolynomialDecay or InverseTimeDecay 

solver = 'Adam'
activation = 'relu' # FIX!!! e.g. sigmoid or relu
dropout = 0 # 0 if no dropout, else fraction of dropout units (e.g. 0.2)   # FIX!!!
batch_normalization = False

weight_init = tfi.glorot_normal() # FIX!!! default: glorot_uniform(); e.g. glorot_normal(), he_normal(), he_uniform(), lecun_normal(), lecun_uniform(), RandomNormal(), RandomUniform(), Zeros() etc.
bias_init = tfi.Zeros() # FIX!!! default: Zeros(); for some possible values see weight initializers

regularization_weight = 0.0 # 0 for no regularization or e.g. 0.01 to apply regularization
regularizer = tfr.l1(l=regularization_weight) # or l2 or l1_l2; used for both weights and biases

num_epochs = 3 # FIX !!!
batch_size = 10 # FIX !!! 

# Sequential network structure.
model = Sequential()

if len(num_hidden) == 0:
  print("Error: Must at least have one hidden layer!")
  sys.exit()  

# add first hidden layer connecting to input layer

model.add(Dense(num_hidden[0], input_dim=num_inputs, activation=activation, kernel_initializer=weight_init, bias_initializer = bias_init, kernel_regularizer=regularizer, bias_regularizer=regularizer))

# if dropout: # dropout at input layer is generally not recommended
#  # dropout of fraction dropout of the neurons and activation layer.
#  model.add(Dropout(dropout))
# #  model.add(Activation("linear"))

if batch_normalization:
  model.add(BatchNormalization())

# potentially further hidden layers
for i in range(1, len(num_hidden)):
  # add hidden layer with len[i] neurons
  model.add(Dense(num_hidden[i], activation=activation, kernel_initializer=weight_init, bias_initializer = bias_init, kernel_regularizer=regularizer, bias_regularizer=regularizer))
#  model.add(Activation("linear"))
  
  if dropout:
  # dropout of fraction dropout of the neurons and activation layer.
    model.add(Dropout(dropout))
  #  model.add(Activation("linear"))

  if batch_normalization:
    model.add(BatchNormalization())  

# output layer
model.add(Dense(units=num_outputs, name = "output", kernel_initializer=weight_init, bias_initializer = bias_init, kernel_regularizer=regularizer, bias_regularizer=regularizer))

if dropout:
# dropout of fraction dropout of the neurons and activation layer.
  model.add(Dropout(dropout))
#  model.add(Activation("linear"))
  
# print configuration
print("\nModel configuration: ")
print(model.get_config())
print("\n")
print("... number of layers: %d" % len(model.layers))

# show how the model looks
model.summary()
      
# compile model
if solver == 'SGD':
  momentum = 0 # e.g. 0.0, 0.5, 0.9 or 0.99
  nesterov = False
  opt = SGD(learning_rate=lr_schedule, momentum=momentum, nesterov=nesterov) # SGD or Adam, Nadam, Adadelta, Adagrad, RMSProp, potentially setting more parameters
elif solver == 'Adam':
  opt = Adam(learning_rate=lr_schedule)
elif solver == 'Nadam':
  opt = Adam(learning_rate=lr_schedule)
elif solver == 'Adadelta':
  opt = Adam(learning_rate=lr_schedule)
elif solver == 'Adagrad':
  opt = Adam(learning_rate=lr_schedule)
elif solver == 'RMSprop':
  opt = RMSprop(learning_rate=lr_schedule)
model.compile(optimizer=opt,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['sparse_categorical_accuracy'])

# histogram of weights (first layer) after initialization
weights = model.layers[0].get_weights()[0]
biases = model.layers[0].get_weights()[1]

nBins = 100
fig, axes = plt.subplots(1, 2, figsize=(15,10))
axes[0].hist(weights.flatten(), nBins)
axes[0].set_xlabel("weights")
axes[0].set_ylabel("counts")
axes[0].set_title("weight histogram after initialization")

axes[1].hist(biases.flatten(), nBins)
axes[1].set_xlabel("biases")
axes[1].set_ylabel("counts")
axes[1].set_title("bias histogram after initialization")
plt.show()

# visualize the weights between input layer and some 
# of the hidden neurons of the first hidden layer after initialization
# model.layers[0].get_weights()[0] is a (784 x numHiddenNeurons) array
# model.layers[0].get_weights()[0].T (transpose) is a (numHiddenNeurons x 784) array,
# the first entry of which contains the weights of all inputs connecting
# to the first hidden neuron; those weights will be displayed in (28 x 28) format
# until all plots (4 x 4, i.e. 16) are "filled" or no more hidden neurons are left
# print("Visualization of the weights between input and some of the hidden neurons of the first hidden layer:")
# fig, axes = plt.subplots(4, 4, figsize=(15,15))
# # use global min / max to ensure all weights are shown on the same scale
# weights = model.layers[0].get_weights()[0]
# vmin, vmax = weights.min(), weights.max()
# for coef, ax in zip(weights.T, axes.ravel()):
#     ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
#                vmax=.5 * vmax)
#     ax.set_xticks(())
#     ax.set_yticks(())

# plt.show()

# Training
history = model.fit(training_input, training_target, epochs=num_epochs, batch_size=batch_size, shuffle="True", verbose=2)

# plot training loss and accuracy 
plt.plot(history.history['loss'], color = 'blue', label = 'training loss')
plt.plot(history.history['sparse_categorical_accuracy'], color = 'red', label = 'traning accuracy')
plt.xlabel('Epoch number')
plt.ylim(0, 1)
plt.legend()
plt.show()

# model evaluation
train_loss = history.history['loss'][num_epochs-1] 
train_acc = history.history['sparse_categorical_accuracy'][num_epochs-1]
val_loss = model.evaluate(validation_input, validation_target)[0]
val_acc = model.evaluate(validation_input, validation_target)[1]
test_loss = model.evaluate(test_input, test_target)[0]
test_acc = model.evaluate(test_input, test_target)[1]

print("\n")
print("final training loss: %f" % train_loss)
print("final training accuracy: %f" % train_acc)
print("final validation loss: %f" % val_loss)
print("final validation accuracy: %f" % val_acc)
print("final test loss: %f" % test_loss)
print("final test accuracy: %f" % test_acc)
print("\n")

# histogram of weights (first layer) after training
weights = model.layers[0].get_weights()[0]
biases = model.layers[0].get_weights()[1]

nBins = 100
fig, axes = plt.subplots(1, 2, figsize=(15,10))
axes[0].hist(weights.flatten(), nBins)
axes[0].set_xlabel("weights")
axes[0].set_ylabel("counts")
axes[0].set_title("weight histogram after training")

axes[1].hist(biases.flatten(), nBins)
axes[1].set_xlabel("biases")
axes[1].set_ylabel("counts")
axes[1].set_title("bias histogram after training")
plt.show()

# visualize the weights between input layer and some 
# of the hidden neurons of the first hidden layer after training
# model.layers[0].get_weights()[0] is a (784 x numHiddenNeurons) array
# model.layers[0].get_weights()[0].T (transpose) is a (numHiddenNeurons x 784) array,
# the first entry of which contains the weights of all inputs connecting
# to the first hidden neuron; those weights will be displayed in (28 x 28) format
# until all plots (4 x 4, i.e. 16) are "filled" or no more hidden neurons are left
print("Visualization of the weights between input and some of the hidden neurons of the first hidden layer:")
fig, axes = plt.subplots(4, 4, figsize=(15,15))
# use global min / max to ensure all weights are shown on the same scale
weights = model.layers[0].get_weights()[0]
vmin, vmax = weights.min(), weights.max()
for coef, ax in zip(weights.T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()