from rcclassifier.ReservoirComputer import ReservoirComputer as rc
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import sys
import os
import time
import pickle


def get_params(num_properties, num_classes, training_length, reservoir_size=10):
    """

    Parameters
    ----------
    num_properties: int
        number of properties per input vector
    num_classes: int
        number of classes that are present
    training_length: int
        number of input vectors
    reservoir_size: int
        size of reservoir

    Returns
    -------
    resparams: dictionary containing the properties for each layer
    """

    # declare parameters
    input_params = {
        "num_properties": num_properties,
        "reservoir_size": reservoir_size,
        # "input_scale":
    }
    reservoir_params = {
        "reservoir_size": reservoir_size
        # "degree": ,
        # "radius": ,
        # "recall":
    }
    output_params = {
        "num_classes": num_classes,
        "training_length": training_length,
        "reservoir_size": reservoir_size
    }

    resparams = {"input_params": input_params,
                 "reservoir_params": reservoir_params,
                 "output_params": output_params}
    return resparams


def format_classes(old_classes):
    """

    Parameters
    ----------
    old_classes: NumPy array
        contains the un-formatted classes for the data

    Returns
    -------
    new_classes:
        formatted classes for the data
    """

    # determine how many classes are present and how many input vectors there are
    num_classes = np.unique(old_classes).size
    num_inputs = old_classes.size

    # create all zeros matrix of appropriate size
    new_classes = np.zeros((num_inputs, num_classes))

    # iterate through classes and adjust properly
    for idx, class_value in enumerate(old_classes):
        new_classes[idx, class_value] = 1

    return new_classes


start_time = time.process_time()
last_time = start_time

data_folder = "data" + os.path.sep + "classifier_data" + os.path.sep
rccModelPath = "models" + os.path.sep + "classifier" + os.path.sep + "rcc" + os.path.sep

datasets = os.listdir(data_folder)
random_states = [None, 0, 42]

rcc_log_csv_file = "logs" + os.path.sep + "classifier" + os.path.sep + "rcc.log.csv"
rcc_log_file = "logs" + os.path.sep + "classifier" + os.path.sep + "rcc.log"

reservoir_sizes = [50, 100, 200]
random_states = [None, 0, 42]
best_acc = 0

if not os.path.exists(rcc_log_csv_file):
    csv_log_file = open(rcc_log_csv_file, 'w')
    csv_log_file.close()

if not os.path.exists(rcc_log_file):
    log_file = open(rcc_log_file, 'w')
    log_file.close()

for dataset in datasets:
    dataset_name = os.path.splitext(dataset)[0]
    dataset_name = dataset_name.replace('.', ',')
    for random_state in random_states:
        for reservoir_size in reservoir_sizes:
            name = dataset_name + "," + str(random_state) + "," + str(reservoir_size)

            last_time = time.process_time()
            X = pd.read_csv(data_folder + dataset)
            X.drop(['interval_id'], axis=1, inplace=True)
            y_temp = X.pop('rains_next_interval')
            y = pd.get_dummies(y_temp)

            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=0.3)

            # get the parameters
            training_length, num_properties = X_train.shape
            num_classes = y_train.shape[1]  # np.unique(training_classes).size

            properties = {
                "reservoir_size": reservoir_size,
                "training_length": training_length,
                "num_properties": num_properties,
                "num_classes": num_classes
            }
            resparams = get_params(**properties)  # parameters for the reservoir layers

            # generate the reservoir computer
            classifier = rc(resparams)

            # train the classifier
            classifier.train_reservoir(X_train.values, y_train.values)

            # test on training data
            output_training = classifier.rc_classification(X_train.values)

            # test on test data
            output_test = classifier.rc_classification(X_test.values)

            acc = (y_test.values == output_test) * 1
            accuracy = np.mean(acc, axis=0)[0]
            print(name + " => " + str(accuracy))

            csv_log_file = open(rcc_log_csv_file, 'a')
            csv_log_file.write(name + "," + str(time.process_time() - last_time) + "," + str(accuracy) + "\n")
            csv_log_file.close()

            if accuracy > best_acc:
                best_acc = accuracy
                print(classification_report(y_test.values, output_test))

            log_file = open(rcc_log_file, 'a')
            log_file.write("Prediction results for dataset: " + name + "\n")
            log_file.write("\nClassifier performance on training dataset\n")
            log_file.write(classification_report(y_train.values, output_training))
            log_file.write("\n\nClassifier performance on test dataset\n")
            log_file.write(classification_report(y_test.values, output_test))
            log_file.write("\nRunning time: " + str(time.process_time() - last_time) + "sec\n\n")
            log_file.close()

            pickle.dump(classifier, open(rccModelPath + name, "wb"))
            print("Running time: " + str(time.process_time() - last_time) + "sec")
            print('#' * 80)

print("Final Running time: " + str(time.process_time() - start_time) + "sec")
