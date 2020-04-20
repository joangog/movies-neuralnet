import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide TensorFlow warnings

import keras.backend as k
from keras import optimizers
from keras.callbacks.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l1
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

N = 943  # users
M = 1682  # movies

folds = 5  # folds of cross validation
hid_neurons_1 = 800 # neurons in hidden layer 1
hid_neurons_2 = 500 # neurons in hidden layer 2
hid_neurons_3 = 200 # neurons in hidden layer 3
epochs = 500
batch = 100
lr = 0.1
momentum = 0.6


#variables to create the average diagram
rmse_list = []
mae_list = []
loss_list = []
val_loss_list = []
min_epochs =500 # epochs that will show on the diagram

# separate inputs(X) and outputs(Y) from dataset
dataset = np.genfromtxt('dataset.data', delimiter='\t', dtype='float')
X = dataset[:, 0:N]
Y = dataset[:, N:N + M]

# create k-fold cross validation model with fold shuffle
kfold = KFold(n_splits=folds, shuffle=True)

#create early stopping callback
es = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=1, mode='min', baseline=None, restore_best_weights=True)

# start cross validation
for fold, (train, test) in enumerate(kfold.split(X, Y)):  # enumerate returns (i,values[i]) where values = kfold.split(X,Y) = (train,test)

    print(f'Fold number: {str(fold)}')

    # create network architecture one level at a time (sequentially)
    net = Sequential()
    net.add(Dense(hid_neurons_1, input_dim=N, activation='sigmoid'))  # hidden layer 1
    net.add(Dense(hid_neurons_2, input_dim=hid_neurons_1, activation='sigmoid'))  # hidden layer 2
    net.add(Dense(hid_neurons_3, input_dim=hid_neurons_2, activation='sigmoid'))  # hidden layer 3
    net.add(Dense(M, activation='tanh'))  # output layer

    # specify error functions
    def rmse(Y_pred, Y_true):
        return k.sqrt(k.mean(k.square(Y_pred - Y_true)))
    def mae(Y_pred, Y_true):
        return k.mean(k.abs(Y_pred - Y_true))

    #parameters of gradient decent
    sgd = optimizers.SGD(lr=lr, momentum=momentum)  # learning rate, momentum
    net.compile(loss='mean_absolute_error', optimizer=sgd, metrics=[rmse, mae])

    # train network
    history = net.fit(X[train], Y[train], batch_size=batch, epochs=epochs, callbacks=[es], validation_split=0.2, verbose=0)

    # test network
    scores = net.evaluate(X[test], Y[test], batch_size=batch, verbose=0)
    #prediction = net.predict(X[test], batch_size=batch, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)

    # print fold results
    print(f'{net.metrics_names[1]}:{scores[1]}\n{net.metrics_names[2]}:{scores[2]}\n')

    # save results to plot average diagram
    rmse_list.append(scores[1])
    mae_list.append(scores[2])
    min_epochs = len(history.history['loss']) if min_epochs>len(history.history['loss']) else min_epochs #only show the average results from the minimum number of epochs of the folds (because of early stopping epochs vary per fold)
    loss_list.append(history.history['loss'])
    val_loss_list.append(history.history['val_loss'])

# keep only the results of the epochs that have data for all the folds
for i in range(len(loss_list)):
    loss_list[i] = loss_list[i][:min_epochs]
    val_loss_list[i] = val_loss_list[i][:min_epochs]

# plot average results from all folds
loss_list = np.asarray(loss_list)
val_loss_list = np.asarray(val_loss_list)
plt.plot(np.mean(loss_list,0))
plt.plot(np.mean(val_loss_list,0))
plt.title(f'Average loss with {hid_neurons_1},{hid_neurons_2},{hid_neurons_3} neurons in hidden layers 1,2,3')
plt.ylabel('Loss')
plt.xlabel('Epoch')
#plt.ylim(0,0.5)
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# print average results from all folds
print(f'average rmse:{np.mean(rmse_list)}')
print(f'average mae:{np.mean(mae_list)}\n')