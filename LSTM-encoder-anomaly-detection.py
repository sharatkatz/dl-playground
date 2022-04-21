"""
Bearing Failure Anomaly Detection
In this workbook, we use an autoencoder neural network to identify vibrational anomalies 
from sensor readings in a set of bearings. The goal is to be able to predict future 
bearing failures before they happen. The vibrational sensor readings are from the 
NASA Acoustics and Vibration Database. Each data set consists of individual files 
that are 1-second vibration signal snapshots recorded at 10 minute intervals. 
Each file contains 20,480 sensor data points that were obtained by reading the 
bearing sensors at a sampling rate of 20 kHz.
https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
"""

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import seed
import tensorflow as tf
import seaborn as sns
from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers

seed(10)

data_dir = 'data/bearing_data'

#values = []
#for root, dirs, files in os.walk(data_dir):
#    for filename in files:
#        with open(os.path.join(root, filename), "r") as f:
#            single_dat = f.read()
#            lines = single_dat.split('\n')
#            for i, line in enumerate(lines):
#                try:
#                    line_items = [float(elem) for elem in line.split('\t')]
#                    values.append(line_items)
#                except ValueError:
#                    pass
#
#merged_data = pd.DataFrame(values, columns=["Bearing 1", "Bearing 2", "Bearing 3", "Bearing 4"])
#
## export data for later use and save time of 
## reading all the files all over again
#merged_data.to_csv('data/bearing_data/merged_data.csv', index=False)

# import back merged data
merged_data = pd.read_csv(os.path.join(data_dir, 'merged_data.csv')) 
train = merged_data[:500]
test = merged_data[500:601]

fig, ax = plt.subplots(figsize=(11, 6), dpi=80)
ax.plot(train['Bearing 1'], label='Bearing 1', color='blue', linewidth=1)
ax.plot(train['Bearing 2'], label='Bearing 2', color='red', linewidth=1)
ax.plot(train['Bearing 3'], label='Bearing 3', color='green', linewidth=1)
ax.plot(train['Bearing 4'], label='Bearing 4', color='cyan', linewidth=1)
plt.legend(loc='lower left')
ax.set_title('Bearing Sensor Training Data', fontsize=16)
plt.show()

# transforming data from the time domain to the frequency domain using fast Fourier transform
train_fft = np.fft.fft(train)
test_fft = np.fft.fft(test)

# frequencies of the healthy sensor signal
fig, ax = plt.subplots(figsize=(11, 6), dpi=80)
ax.plot(train_fft[:,0].real, label='Bearing 1', color='blue', linewidth=1)
ax.plot(train_fft[:,1].imag, label='Bearing 2', color='red', linewidth=1)
ax.plot(train_fft[:,2].real, label='Bearing 3', color='green', linewidth=1)
ax.plot(train_fft[:,3].real, label='Bearing 4', color='cyan', linewidth=1)
plt.legend(loc='lower left')
ax.set_title('Bearing Sensor Training Frequency Data', fontsize=16)
plt.show()

mean = train_fft.mean()
std = train_fft.std()
X_train = (train_fft - mean) / std
X_test = (test_fft - mean) / std

# reshape inputs for LSTM [samples, timesteps, features]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
print("Training data shape:", X_train.shape)
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
print("Test data shape:", X_test.shape)

# define the autoencoder network model
def autoencoder_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(32, activation='relu', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(16, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(16, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(32, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model

# create the autoencoder model
model = autoencoder_model(X_train)
model.compile(optimizer='adam', loss='mae')
print(model.summary())

# fit the model to the data
nb_epochs = 50
batch_size = 50
history = model.fit(X_train, 
        X_train, 
        epochs=nb_epochs, 
        batch_size=batch_size,
        validation_split=0.05).history

# plot the training losses
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(history['loss'], 'b', label='Train', linewidth=2)
ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
ax.set_title('Model loss', fontsize=16)
ax.set_ylabel('Loss (mae)')
ax.set_xlabel('Epoch')
ax.legend(loc='upper right')
plt.show()

# plot the loss distribution of the training set
X_pred = model.predict(X_train)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=train.columns)
X_pred.index = train.index

scored = pd.DataFrame(index=train.index)
Xtrain = X_train.reshape(X_train.shape[0], X_train.shape[2])
scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtrain), axis = 1)

def histogram_boxplot(data, xlabel=None, title=None, font_scale=2, figsize=(8,6), bins=None):
    """ Boxplot and histogram combined
    data: 1-d data array
    xlabel: xlabel 
    title: title
    font_scale: the scale of the font (default 2)
    figsize: size of fig (default (9,8))
    bins: number of bins (default None / auto)

    example use: histogram_boxplot(np.random.rand(100), bins = 20, title="Fancy plot")
    """

    sns.set(font_scale=font_scale)
    f2, (ax_box2, ax_hist2) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=figsize)
    sns.boxplot(data, ax=ax_box2)
    sns.distplot(data, ax=ax_hist2, bins=bins, kde=True) if bins else sns.distplot(data, ax=ax_hist2)
    if xlabel: ax_hist2.set(xlabel=xlabel)
    if title: ax_box2.set(title=title)
    plt.show()

histogram_boxplot(scored['Loss_mae'],  title='Loss Distribution', bins=200)

# calculate the loss on the test set
X_pred = model.predict(X_test)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=test.columns)
X_pred.index = test.index

scored = pd.DataFrame(index=test.index)
Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtest), axis = 1)
scored['Threshold'] = 1.0
scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']

# calculate the same metrics for the training set 
# and merge all data in a single dataframe for plotting
X_pred_train = model.predict(X_train)
X_pred_train = X_pred_train.reshape(X_pred_train.shape[0], X_pred_train.shape[2])
X_pred_train = pd.DataFrame(X_pred_train, columns=train.columns)
X_pred_train.index = train.index

scored_train = pd.DataFrame(index=train.index)
scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train-Xtrain), axis = 1)
scored_train['Threshold'] = 1.0
scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']
scored = pd.concat([scored_train, scored])

# plot bearing failure time plot
scored.plot(logy=True,  figsize=(16,9), ylim=[1e-2,1e2], color=['blue','red'])
plt.show()
