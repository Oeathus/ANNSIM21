from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import os
import pickle
import time

data_folder = "data" + os.path.sep + "classifier_data" + os.path.sep
svmModelPath = "models" + os.path.sep + "classifier" + os.path.sep + "svm" + os.path.sep

datasets = os.listdir(data_folder)

svm_log_csv_file = "logs" + os.path.sep + "classifier" + os.path.sep + "svm.log.csv"
svm_log_file = "logs" + os.path.sep + "classifier" + os.path.sep + "svm.log"

random_states = [None, 0, 42]
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

start_time = time.process_time()
last_time = start_time

best_acc = 0
best_settings = ""

if not os.path.exists(svm_log_file):
    log_file = open(svm_log_file, 'w')
    log_file.close()

if not os.path.exists(svm_log_csv_file):
    log_file = open(svm_log_csv_file, 'w')
    log_file.close()

for dataset in datasets:
    dataset_name = os.path.splitext(dataset)[0]
    dataset_name = dataset_name.replace('.', ',')
    for random_state in random_states:
        for kernel in kernels:
            model_name = dataset_name + "," + str(random_state) + "," + kernel

            last_time = time.process_time()
            X = pd.read_csv(data_folder + dataset)
            X.drop(['interval_id'], axis=1, inplace=True)
            y = X.pop('rains_next_interval')
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=0.3)

            sv_classifier = SVC(kernel=kernel)
            sv_classifier.fit(X_train, y_train)
            y_pred = sv_classifier.predict(X_test)
            score = sv_classifier.score(X_test, y_test)

            log_text = model_name + "," + str(
                time.process_time() - last_time) + "s\n" + classification_report(y_test, y_pred) + "\n"
            print(model_name + "," + str(score) + "," + str(time.process_time() - last_time))
            if score > best_acc:
                best_acc = score
                best_settings = model_name

            pickle.dump(sv_classifier, open(svmModelPath + model_name, "wb"))

            with open(svm_log_csv_file, 'a') as log_file:
                log_file.write(model_name + "," + str(score) + "\n")
                log_file.close()
            with open(svm_log_file, 'a') as log_file:
                log_file.write(log_text)
                log_file.close()

print("Final Running time: " + str(time.process_time() - start_time) + "sec")
print("Best Accuracy and Settings as Follows:")
print(best_settings)
print(str(best_acc))
