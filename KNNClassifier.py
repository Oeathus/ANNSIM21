import copy
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import os
import pickle

data_folder = "data" + os.path.sep + "classifier_data" + os.path.sep
knnModelPath = "models" + os.path.sep + "classifier" + os.path.sep + "knn" + os.path.sep

datasets = os.listdir(data_folder)
random_states = [None, 0, 42]

knn_log_csv_file = "logs" + os.path.sep + "classifier" + os.path.sep + "knn.log.csv"
knn_log_file = "logs" + os.path.sep + "classifier" + os.path.sep + "knn.log"

start_time = time.time()
start_process_time = time.process_time()
last_time = start_time
last_process_time = start_process_time

if not os.path.exists(knn_log_file):
    log_file = open(knn_log_file, 'w')
    log_file.close()

if not os.path.exists(knn_log_csv_file):
    log_file = open(knn_log_csv_file, 'w')
    log_file.close()

for dataset in datasets:
    dataset_name = os.path.splitext(dataset)[0]
    dataset_name = dataset_name.replace('.', ',')
    for random_state in random_states:
        model_name = dataset_name + "," + str(random_state)
        strongest_neighbor = 0
        best_score = 0
        best_classification = ""
        best_knn = None

        inner_log_file = "logs" + os.path.sep + "classifier" + os.path.sep + "inner_knn_log" + os.path.sep + model_name + ".csv"
        if not os.path.exists(inner_log_file):
            log_file = open(inner_log_file, 'w')
            log_file.close()

        X = pd.read_csv(data_folder + dataset)
        X.drop(['interval_id'], axis=1, inplace=True)
        y = X.pop('rains_next_interval')
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=0.3)

        last_inner_time = time.time()
        last_inner_process_time = time.process_time()
        for i in range(1, 26):
            last_inner_time = time.time()
            last_inner_process_time = time.process_time()
            knn = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            score = knn.score(X_test, y_test)
            if score > best_score:
                best_knn = copy.deepcopy(knn)
                best_score = score
                strongest_neighbor = i
                best_classification = classification_report(y_test, y_pred)
            print(model_name + "," + str(i) + " => " + str(
                score) + "\nInner Running time: " + str(time.time() - last_inner_time) + "sec\n" +
                  "Inner Process Running time: " + str(time.process_time() - last_inner_process_time) + "sec\n")
            with open(inner_log_file, 'a') as log_file:
                log_file.write(model_name + "," + str(score) + "," +
                               str(time.process_time() - last_inner_process_time) + "\n")
                log_file.close()
        pickle.dump(best_knn, open(knnModelPath + model_name + "," + str(strongest_neighbor) + ".knn", "wb"))

        log_text = model_name + "," + str(strongest_neighbor) + "\nRunning time: " + str(
            time.time() - last_time) + "sec\n" + "\nRunning Process time: " + str(
            time.process_time() - last_process_time) + "sec\n" + best_classification + "\n"
        print(log_text)
        with open(knn_log_csv_file, 'a') as log_file:
            log_file.write(model_name + "," + str(strongest_neighbor) + "," + str(best_score) + "\n")
            log_file.close()
        with open(knn_log_file, 'a') as log_file:
            log_file.write(log_text)
            log_file.close()
        last_time = time.time()
        last_process_time = time.process_time()
print("Final Running time: " + str(time.time() - start_time) + "sec")
print("Final Process Running time: " + str(time.process_time() - start_process_time) + "sec")
