import copy
import time
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import pandas as pd
import os
import pickle


data_folder = "data" + os.path.sep + "regressor_data" + os.path.sep
knnModelPath = "models" + os.path.sep + "regressor" + os.path.sep + "knn" + os.path.sep

datasets = os.listdir(data_folder)
random_states = [None, 0, 42]

knn_log_csv_file = "logs" + os.path.sep + "regressor" + os.path.sep + "knn.log.csv"
knn_log_file = "logs" + os.path.sep + "regressor" + os.path.sep + "knn.log"

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

if not os.path.exists(knn_log_csv_file):
    log_file = open(knn_log_csv_file, 'w')
    log_file.close()

for dataset in datasets:
    dataset_name = os.path.splitext(dataset)[0]
    dataset_name = dataset_name.replace('.', ',')
    for random_state in random_states:
        model_name = dataset_name + "," + str(random_state)
        best_knn = None
        bestInnerR2 = 0
        bestInnerR2MSE = 1
        bestInnerR2RMSE = 1
        bestInnerR2Corr = 0
        bestInnerR2Model = ""

        inner_log_file = "logs" + os.path.sep + "regressor" + os.path.sep + "inner_knn_log" + os.path.sep + model_name + ".csv"
        if not os.path.exists(inner_log_file):
            log_file = open(inner_log_file, 'w')
            log_file.close()

        last_time = time.process_time()
        X = pd.read_csv(data_folder + dataset)
        X.drop(['interval_id'], axis=1, inplace=True)
        y = X.pop('next_p01i')
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=0.3)

        for i in range(1, 1001):
            last_inner_process_time = time.process_time()
            knn = KNeighborsRegressor(n_neighbors=i, n_jobs=-1)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)

            r_squared = r2_score(y_test, y_pred)  # Best is 1.0
            skMSE = mean_squared_error(y_test, y_pred, squared=True)  # Best is 0.0
            skRMSE = mean_squared_error(y_test, y_pred, squared=False)  # Best is 0.0
            sciPearson = pearsonr(y_test, y_pred)[0]  # Best is 1.0
            finish_time = time.process_time() - last_inner_process_time

            if r_squared > bestInnerR2:
                bestInnerR2 = r_squared
                bestInnerR2MSE = skMSE
                bestInnerR2RMSE = skRMSE
                bestInnerR2Corr = sciPearson
                bestInnerR2Model = model_name + "," + str(i)
                best_knn = copy.deepcopy(knn)
            if r_squared > bestR2:
                bestR2 = r_squared
                bestR2Model = model_name + "," + str(i)
            if skMSE < bestMSE:
                bestMSE = skMSE
                bestMSEModel = model_name + "," + str(i)
            if skRMSE < bestRMSE:
                bestRMSE = skRMSE
                bestRMSEModel = model_name + "," + str(i)
            if sciPearson > bestCorr:
                bestCorr = sciPearson
                bestCorrModel = model_name + "," + str(i)

            print(model_name + "," + str(i) + " => " + str(r_squared) +
                  "\nInner Running time: " + str(finish_time) + "sec\n")

            with open(inner_log_file, 'a') as log_file:
                log_file.write(model_name + "," + str(i) + "," +
                               str(r_squared) + "," +
                               str(skMSE) + "," +
                               str(skRMSE) + "," +
                               str(sciPearson) + "," +
                               str(finish_time) + "\n")
                log_file.close()

        finish_time = time.process_time() - last_time
        pickle.dump(best_knn, open(knnModelPath + bestInnerR2Model + ".knn", "wb"))

        print(bestInnerR2Model + " => R2: " + str(bestInnerR2) + "\nRunning time: " + str(finish_time) + "sec\n")

        with open(knn_log_csv_file, 'a') as log_file:
            log_file.write(bestInnerR2Model + "," +
                           str(bestInnerR2) + "," +
                           str(bestInnerR2MSE) + "," +
                           str(bestInnerR2RMSE) + "," +
                           str(bestInnerR2Corr) + "\n")
            log_file.close()

print("Running Time: " + str(time.process_time() - start_time))
print("Best R^2 was " + str(bestR2) + " from model: " + str(bestR2Model))
print("Best MSE was " + str(bestMSE) + " from model: " + str(bestMSEModel))
print("Best RMSE was " + str(bestRMSE) + " from model: " + str(bestRMSEModel))
print("Best Corr was " + str(bestCorr) + " from model: " + str(bestCorrModel))
