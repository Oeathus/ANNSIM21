import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import os
import time


def wx_input_fn(features, targets=None, num_epochs=None, shuffle=True, batch_size=128):
    return tf.compat.v1.estimator.inputs.pandas_input_fn(x=features, y=targets,
                                                         num_epochs=num_epochs,
                                                         shuffle=shuffle,
                                                         batch_size=batch_size)


data_folder = "data" + os.path.sep + "regressor_data" + os.path.sep
dnnModelPath = "models" + os.path.sep + "regressor" + os.path.sep + "dnn" + os.path.sep
datasets = os.listdir(data_folder)
random_states = [None, 0, 42]
layers = [2, 3, 4, 5, 10, 20, 30]
dnnr_log_csv_file = "logs" + os.path.sep + "regressor" + os.path.sep + "dnn.log.csv"

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

if not os.path.exists(dnnr_log_csv_file):
    log_file = open(dnnr_log_csv_file, 'w')
    log_file.close()

for dataset in datasets:
    dataset_name = os.path.splitext(dataset)[0]
    dataset_name = dataset_name.replace('.', ',')
    for layer_count in layers:
        for random_state in random_states:
            last_time = time.process_time()
            name = dataset_name + "," + str(random_state) + "," + str(layer_count)

            model_folder = dnnModelPath + name
            if not os.path.exists(model_folder):
                os.mkdir(model_folder)

            X = pd.read_csv(data_folder + dataset)
            X.drop(['interval_id'], axis=1, inplace=True)
            y = X.pop('next_p01i')
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=0.3)

            feature_cols = [tf.feature_column.numeric_column(col) for col in X.columns]
            inner_layers = [X_train.shape[1]] * layer_count
            regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                                  hidden_units=inner_layers,
                                                  model_dir=model_folder)
            print("Training " + name + " on " + str(X_train.shape[0]) + " rows ...")
            regressor.train(input_fn=wx_input_fn(X_train, targets=y_train), steps=400)
            print("Predicting ...")
            y_pred_gen = regressor.predict(input_fn=wx_input_fn(X_test, shuffle=False))
            i = 0
            y_pred = []
            for pred in y_pred_gen:
                y_pred.append(pred["predictions"][0])
                i += 1
                if i >= X_test.shape[0]:
                    break
            print(y_test.values)
            print(y_pred)
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

            with open(dnnr_log_csv_file, 'a') as log_file:
                log_file.write(name + "," +
                               str(r_squared) + "," +
                               str(skMSE) + "," +
                               str(skRMSE) + "," +
                               str(sciPearson) + "," +
                               str(finish_time) + "\n")
                log_file.close()
            print("Finished " + name + " in " + str(finish_time))

print("Running Time: " + str(time.time() - start_time))
print("Best R^2 was " + str(bestR2) + " from model: " + str(bestR2Model))
print("Best MSE was " + str(bestMSE) + " from model: " + str(bestMSEModel))
print("Best RMSE was " + str(bestRMSE) + " from model: " + str(bestRMSEModel))
print("Best Corr was " + str(bestCorr) + " from model: " + str(bestCorrModel))
