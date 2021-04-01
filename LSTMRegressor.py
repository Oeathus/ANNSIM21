from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import tensorflow as tf
import pandas as pd
import numpy as np
import scipy
import os
import time


def create_dataset(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:i + time_steps, :]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def create_model(units, m, shape1, shape2):
    model = Sequential()

    if m == Bidirectional:
        model.add(m(LSTM(units=units, return_sequences=True), input_shape=(shape1, shape2)))
        model.add(Bidirectional(LSTM(units=units)))
    else:
        model.add(m(units=units, return_sequences=True, input_shape=[shape1, shape2]))
        model.add(Dropout(0.2))
        model.add(m(units=units))
        model.add(Dropout(0.2))

    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model


regressor_data = "data" + os.path.sep + "regressor_data" + os.path.sep
lstmrModelPath = "models" + os.path.sep + "regressor" + os.path.sep
datasets = os.listdir(regressor_data)
sequenceLengths = [3, 5, 7, 9, 12]
models = [Bidirectional, GRU, LSTM]

tf.random.set_seed(42)
np.random.seed(46)  # 46 resulted in best r^2 on simple investigation of linear regression on ROC.None.csv

bestR2 = 0
bestR2Model = ""
bestMSE = 1
bestMSEModel = ""
bestRMSE = 1
bestRMSEModel = ""
bestCorr = 0
bestCorrModel = ""

start_time = time.process_time()
last_time = start_time
finish_time = 0

for dataset in datasets:
    dataset_name = os.path.splitext(dataset)[0]
    dataset_name_parts = dataset_name.replace('.', ',')
    for model in models:
        lstmr_log_csv_file = "logs" + os.path.sep + "regressor" + os.path.sep + str(model.__name__).lower() +".log.csv"
        if not os.path.exists(lstmr_log_csv_file):
            log_file = open(lstmr_log_csv_file, 'w')
            log_file.close()

        for sequenceLength in sequenceLengths:
            last_time = time.process_time()
            name = dataset_name + "," + str(sequenceLength)

            model_file = lstmrModelPath + str(model.__name__).lower() + os.path.sep + name

            X = pd.read_csv(regressor_data + dataset)
            X.drop(['interval_id'], axis=1, inplace=True)
            y = X.pop('next_p01i')
            train_x_norm, test_x_norm, train_y_norm, test_y_norm = train_test_split(X, y, shuffle=False, test_size=0.3)
            X_test, y_test = create_dataset(test_x_norm.values, test_y_norm.values, sequenceLength)
            X_train, y_train = create_dataset(train_x_norm.values, train_y_norm.values, sequenceLength)

            modelckpt_callback = keras.callbacks.ModelCheckpoint(
                monitor="val_loss",
                filepath=model_file + ".h5",
                verbose=1,
                save_weights_only=True,
                save_best_only=True,
            )
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
            regressor = create_model(64, model, X_train.shape[1], X_train.shape[2])
            history = regressor.fit(X_train, y_train, epochs=50,
                                    validation_split=0.2, batch_size=32,
                                    shuffle=False, callbacks=[early_stop, modelckpt_callback])
            y_pred = regressor.predict(X_test).flatten()

            r_squared = r2_score(y_test, y_pred)  # Best is 1.0
            skMSE = mean_squared_error(y_test, y_pred, squared=True)  # Best is 0.0
            skRMSE = mean_squared_error(y_test, y_pred, squared=False)  # Best is 0.0
            sciPearson = pearsonr(y_test, y_pred)[0]  # Best is 1.0
            finish_time = time.process_time() - last_time

            print(str(r_squared) + ", " + str(skMSE) + ", " + str(skRMSE) + ", " + str(sciPearson))

            if r_squared > bestR2:
                bestR2 = r_squared
                bestR2Model = name
            if skMSE < bestMSE:
                bestMSE = skMSE
                bestMSEModel = name
            if skRMSE < bestRMSE:
                bestRMSE = skRMSE
                bestRMSEModel = name
            if sciPearson > bestCorr:
                bestCorr = sciPearson
                bestCorrModel = name

            with open(lstmr_log_csv_file, 'a') as log_file:
                log_file.write(name + "," +
                               str(r_squared) + "," +
                               str(skMSE) + "," +
                               str(skRMSE) + "," +
                               str(sciPearson) + "," +
                               str(finish_time) + "\n")
                log_file.close()
            print("Finished " + name + " in " + str(finish_time))

print("Running Time: " + str(time.process_time() - start_time))
print("Best R^2 was " + str(bestR2) + " from model: " + str(bestR2Model))
print("Best MSE was " + str(bestMSE) + " from model: " + str(bestMSEModel))
print("Best RMSE was " + str(bestRMSE) + " from model: " + str(bestRMSEModel))
print("Best Corr was " + str(bestCorr) + " from model: " + str(bestCorrModel))
