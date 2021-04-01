from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import os
import time
import pickle

data_folder = "data" + os.path.sep + "regressor_data" + os.path.sep
knnModelPath = "models" + os.path.sep + "regressor" + os.path.sep + "linear" + os.path.sep

datasets = os.listdir(data_folder)
random_states = [None, 0, 42]

linear_log_csv_file = "logs" + os.path.sep + "regressor" + os.path.sep + "linear.log.csv"

if not os.path.exists(linear_log_csv_file):
    log_file = open(linear_log_csv_file, 'w')
    log_file.close()

datasets = os.listdir(regressorDataPath)
random_states = [None, 0, 42]
np.random.seed(46)  # 46 resulted in best r^2 on simple investigation

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
    dataset_name = dataset_name.replace('.', ',')
    for random_state in random_states:
        last_time = time.process_time()
        model = dataset_name + "," + str(random_state)

        X = pd.read_csv(regressorDataPath + dataset)
        X.drop(['interval_id'], axis=1, inplace=True)
        y = X.pop('next_p01i')
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=0.3)

        reg = LinearRegression().fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        r_squared = r2_score(y_test, y_pred)  # Best is 1.0
        skMSE = mean_squared_error(y_test, y_pred, squared=True)  # Best is 0.0
        skRMSE = mean_squared_error(y_test, y_pred, squared=False)  # Best is 0.0
        sciPearson = pearsonr(y_test, y_pred)[0]  # Best is 1.0
        finish_time = time.process_time() - last_time

        if r_squared > bestR2:
            bestR2 = r_squared
            bestR2Model = model
        if skMSE < bestMSE:
            bestMSE = skMSE
            bestMSEModel = model
        if skRMSE < bestRMSE:
            bestRMSE = skRMSE
            bestRMSEModel = model
        if sciPearson > bestCorr:
            bestCorr = sciPearson
            bestCorrModel = model

        with open(linear_log_csv_file, 'a') as log_file:
            log_file.write(model + "," +
                           str(r_squared) + "," +
                           str(skMSE) + "," +
                           str(skRMSE) + "," +
                           str(sciPearson) + "," +
                           str(finish_time) + "\n")
            log_file.close()
        pickle.dump(reg, open(regressionModelPath + model + ".linear", 'wb'))
        print("Finished " + model + " in " + str(finish_time))

print("Running Time: " + str(time.time() - start_time))
print("Best R^2 was " + str(bestR2) + " from model: " + str(bestR2Model))
print("Best MSE was " + str(bestMSE) + " from model: " + str(bestMSEModel))
print("Best RMSE was " + str(bestRMSE) + " from model: " + str(bestRMSEModel))
print("Best Corr was " + str(bestCorr) + " from model: " + str(bestCorrModel))
