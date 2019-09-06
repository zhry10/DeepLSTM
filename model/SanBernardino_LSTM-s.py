import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io
import keras
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.optimizers import RMSprop, Adam
from keras.layers import LSTM, Activation, CuDNNLSTM
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import time
from random import shuffle
from sklearn.externals import joblib  # save scaler

# Setup GPU for training (use tensorflow v1.9 for CuDNNLSTM)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # CPU:-1; GPU0: 1; GPU1: 0;


def Generate_data(X_data0, y_data0, window_size=50):
    X_new_temp = []
    y_new_temp = []
    for ii in range(len(X_data0)):
        X_temp = X_data0[ii]
        y_temp = y_data0[ii]
        X_new = []
        y_new = []
        for jj in range(int(np.floor(len(X_temp) / window_size))):
            X_new.append(X_temp[jj * window_size:(jj + 1) * window_size])
            y_new.append(y_temp[(jj + 1) * window_size - 1, :])
            # y_new.append(y_temp[(jj + 1) * window_size - 1])

        X_new_temp.append(np.array(X_new))
        y_new_temp.append(np.array(y_new))

    X_data_new0 = np.array(X_new_temp)
    y_data_new0 = np.array(y_new_temp)

    return X_data_new0, y_data_new0


# Load data
dataDir = 'C:/Users/sunsh/OneDrive/Desktop/CAS Paper/'  # Replace the directory
mat = scipy.io.loadmat(dataDir+'data/data_SanBernardino.mat')

X_data = mat['input_tf']
y_data = mat['target_tf']
train_indices = mat['trainInd'] - 1
test_indices = mat['valInd'] - 1

X_data = np.reshape(X_data, [X_data.shape[0], X_data.shape[1], 1])

# Scale data
X_data_flatten = np.reshape(X_data, [X_data.shape[0]*X_data.shape[1], 1])
scaler_X = MinMaxScaler(feature_range=(-1, 1))
scaler_X.fit(X_data_flatten)
X_data_flatten_map = scaler_X.transform(X_data_flatten)
X_data_map = np.reshape(X_data_flatten_map, [X_data.shape[0], X_data.shape[1], 1])

y_data_flatten = np.reshape(y_data, [y_data.shape[0]*y_data.shape[1], y_data.shape[2]])
scaler_y = MinMaxScaler(feature_range=(-1, 1))
scaler_y.fit(y_data_flatten)
y_data_flatten_map = scaler_y.transform(y_data_flatten)
y_data_map = np.reshape(y_data_flatten_map, [y_data.shape[0], y_data.shape[1], y_data.shape[2]])

# Unknown data
X_pred = mat['input_pred_tf']
y_pred_ref = mat['target_pred_tf']
X_pred = np.reshape(X_pred, [X_pred.shape[0], X_pred.shape[1], 1])

# Scale data
X_pred_flatten = np.reshape(X_pred, [X_pred.shape[0]*X_pred.shape[1], 1])
X_pred_flatten_map = scaler_X.transform(X_pred_flatten)
X_pred_map = np.reshape(X_pred_flatten_map, [X_pred.shape[0], X_pred.shape[1], 1])

y_pred_ref_flatten = np.reshape(y_pred_ref, [y_pred_ref.shape[0]*y_pred_ref.shape[1], y_pred_ref.shape[2]])
y_pred_ref_flatten_map = scaler_y.transform(y_pred_ref_flatten)
y_pred_ref_map = np.reshape(y_pred_ref_flatten_map, [y_pred_ref.shape[0], y_pred_ref.shape[1], y_pred_ref.shape[2]])

windowsize = 2
X_data_new, y_data_new = Generate_data(X_data_map, y_data_map, windowsize)

X_data_new = np.reshape(X_data_new, [X_data_new.shape[0], X_data_new.shape[1], X_data_new.shape[2]])
# y_data = np.reshape(y_data, [y_data.shape[0], y_data.shape[1], 1])

X_train = X_data_new[0:len(train_indices[0])]
y_train = y_data_new[0:len(train_indices[0])]
X_test = X_data_new[len(train_indices[0]):]
y_test = y_data_new[len(train_indices[0]):]

X_pred, y_pred_ref = Generate_data(X_pred_map, y_pred_ref_map, windowsize)
X_pred = np.reshape(X_pred, [X_pred.shape[0], X_pred.shape[1], X_pred.shape[2]])

# n_epoch = 200
data_dim = X_train.shape[2]  # number of input features
timesteps = X_train.shape[1]
num_classes = y_train.shape[2]  # number of output features
batch_size = 2

rms = RMSprop(lr=0.001, decay=0.0001)
adam = Adam(lr=0.001, decay=0.0001)
model = Sequential()
model.add(CuDNNLSTM(30, return_sequences=True, stateful=False, input_shape=(None, data_dim)))
model.add(Activation('relu'))
# model.add(Dropout(0.2))
model.add(CuDNNLSTM(30, return_sequences=True, stateful=False))
model.add(Activation('relu'))
# model.add(Dropout(0.2))
model.add(Dense(30))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.summary()

model.compile(loss='mean_squared_error',  # categorical_crossentropy, mean_squared_error, mean_absolute_error
              optimizer=adam,  # RMSprop(), Adagrad, Nadam, Adagrad, Adadelta, Adam, Adamax,
              metrics=['mse'])
best_loss = 100
train_loss = []
test_loss = []
history = []

with tf.device('/device:GPU:0'):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    session = tf.Session(config=config)
    # tf.Session(config=tf.ConfigProto(log_device_placement=True))

    start = time.time()

    epochs = 20000
    for e in range(epochs):
        print('epoch = ', e + 1)

        Ind = list(range(len(X_data_new)))
        shuffle(Ind)
        ratio_split = 0.7
        Ind_train = Ind[0:round(ratio_split*len(X_data_new))]
        Ind_test = Ind[round(ratio_split*len(X_data_new)):]

        X_train = X_data_new[Ind_train]
        y_train = y_data_new[Ind_train]
        X_test = X_data_new[Ind_test]
        y_test = y_data_new[Ind_test]

        model.fit(X_train, y_train,
                  batch_size=batch_size,
                  # validation_split=0.3,
                  validation_data=(X_test, y_test),
                  shuffle=True,
                  epochs=1)
        score0 = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=0)
        score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
        train_loss.append(score0[0])
        test_loss.append(score[0])

        # Save the best trained model with minimum testing loss
        if test_loss[e] < best_loss:
            best_loss = test_loss[e]
            model.save(dataDir + 'results/SanBernardino/my_best_model.h5')

    end = time.time()
    running_time = (end - start)/3600
    print('Running Time: ', running_time, ' hour')

# Plot training and testing loss
plt.figure()
plt.plot(np.array(train_loss), 'b-')
plt.plot(np.array(test_loss), 'm-')

# plt.figure()
# plt.plot(np.log(np.array(train_loss)), 'b-')
# plt.plot(np.log(np.array(test_loss)), 'm-')

# Load the best model
model_best = load_model(dataDir + 'results/SanBernardino/my_best_model.h5')

X_train = X_data_new[0:len(train_indices[0])]
y_train = y_data_new[0:len(train_indices[0])]
X_test = X_data_new[len(train_indices[0]):]
y_test = y_data_new[len(train_indices[0]):]

y_train_pred = model_best.predict(X_train)
y_test_pred = model_best.predict(X_test)
y_pure_preds = model_best.predict(X_pred)

# Reverse map to original magnitude
X_train_orig = X_data[0:len(train_indices[0])]
y_train_orig = y_data[0:len(train_indices[0])]
X_test_orig = X_data[len(train_indices[0]):]
y_test_orig = y_data[len(train_indices[0]):]
X_pred_orig = mat['input_pred_tf']
y_pred_ref_orig = mat['target_pred_tf']

y_train_pred_flatten = np.reshape(y_train_pred, [y_train_pred.shape[0]*y_train_pred.shape[1], y_train_pred.shape[2]])
y_train_pred = scaler_y.inverse_transform(y_train_pred_flatten)
y_train_pred = np.reshape(y_train_pred, [y_train.shape[0], y_train.shape[1], y_train.shape[2]])

for sample in range(len(y_train)):
    plt.figure()
    plt.plot(y_train_orig[sample][windowsize::windowsize, 0], label='True')
    plt.plot(y_train_pred[sample][:, 0], label='Predict')
    plt.title('Training')
    plt.legend()

y_test_pred_flatten = np.reshape(y_test_pred, [y_test_pred.shape[0]*y_test_pred.shape[1], y_test_pred.shape[2]])
y_test_pred = scaler_y.inverse_transform(y_test_pred_flatten)
y_test_pred = np.reshape(y_test_pred, [y_test.shape[0], y_test.shape[1], y_test.shape[2]])

for sample in range(len(y_test)):
    plt.figure()
    plt.plot(y_test_orig[sample][windowsize::windowsize, 0], label='True')
    plt.plot(y_test_pred[sample][:, 0], label='Predict')
    plt.title('Testing')
    plt.legend()

y_pure_preds_flatten = np.reshape(y_pure_preds, [y_pure_preds.shape[0]*y_pure_preds.shape[1], y_pure_preds.shape[2]])
y_pure_preds = scaler_y.inverse_transform(y_pure_preds_flatten)
y_pure_preds = np.reshape(y_pure_preds, [y_pred_ref.shape[0], y_pred_ref.shape[1], y_pred_ref.shape[2]])

for sample in range(len(y_pred_ref)):
    plt.figure()
    plt.plot(y_pred_ref_orig[sample][windowsize::windowsize, 0], label='True')
    plt.plot(y_pure_preds[sample][:, 0], label='Predict')
    plt.title('Prediction')
    plt.legend()

# Save scaler
joblib.dump(scaler_X, dataDir+'results/SanBernardino/scaler_X.save')
joblib.dump(scaler_y, dataDir+'results/SanBernardino/scaler_y.save')
# And now to load...
# scaler_X = joblib.load(dataDir+'results/SanBernardino/scaler_X.save')
# scaler_y = joblib.load(dataDir+'results/SanBernardino/scaler_y.save')

scipy.io.savemat(dataDir+'results/SanBernardino/results_SanBernardino.mat',
                 {'y_train': y_train, 'y_train_orig': y_train_orig, 'y_train_pred': y_train_pred,
                  'y_test': y_test, 'y_test_orig': y_test_orig, 'y_test_pred': y_test_pred,
                  'y_pred_ref': y_pred_ref, 'y_pred_ref_orig': y_pred_ref_orig, 'y_pure_preds': y_pure_preds,
                  'X_train': X_train, 'X_test': X_test, 'X_pred': X_pred,
                  'train_indices': train_indices[0], 'test_indices': test_indices[0],
                  'train_loss': train_loss, 'test_loss': test_loss, 'best_loss': best_loss,
                  'windowsize': windowsize, 'running_time': running_time, 'epochs': epochs})
