import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import os
import time


def input_fn(features, labels, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    return dataset.batch(batch_size)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

data_folder = "data" + os.path.sep + "classifier_data" + os.path.sep
dnnModelPath = "models" + os.path.sep + "classifier" + os.path.sep + "dnn" + os.path.sep

datasets = os.listdir(data_folder)
random_states = [None, 0, 42]
layers = [2, 3, 4, 5]
dnn_log_csv_file = "logs" + os.path.sep + "classifier" + os.path.sep + "dnn.log.csv"
dnn_log_file = "logs" + os.path.sep + "classifier" + os.path.sep + "dnn.log"
best_acc = 0
best_settings = ""

start_time = time.process_time()
last_time = start_time

if not os.path.exists(dnn_log_file):
    log_file = open(dnn_log_file, 'w')
    log_file.close()

if not os.path.exists(dnn_log_csv_file):
    log_file = open(dnn_log_csv_file, 'w')
    log_file.close()

for dataset in datasets:
    dataset_name = os.path.splitext(dataset)[0]
    dataset_name = dataset_name.replace('.', ',')
    for layer_count in layers:
        for random_state in random_states:
            last_time = time.process_time()
            model_name = dataset_name + "," + str(random_state) + "," + str(layer_count)

            model_folder = dnnModelPath + model_name
            if not os.path.exists(model_folder):
                os.mkdir(model_folder)

            X = pd.read_csv(data_folder + dataset)
            X.drop(['interval_id'], axis=1, inplace=True)
            y = X.pop('rains_next_interval')
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=0.3)
            classes = len(set(y.values))

            feature_columns = []
            for key in X.keys():
                feature_columns.append(tf.feature_column.numeric_column(key=key))

            inner_layers = [X_train.shape[1]] * layer_count
            Deep = tf.estimator.DNNClassifier(
                feature_columns=feature_columns,
                hidden_units=inner_layers,
                n_classes=classes,
                model_dir=model_folder
            )

            Deep.train(input_fn=lambda: input_fn(X_train, y_train))
            Deep_eval_result = Deep.evaluate(input_fn=lambda: input_fn(X_test, y_test))

            pred_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(X_test, batch_size=256, shuffle=False)
            y_pred_gen = Deep.predict(input_fn=pred_input_fn)
            y_pred = []
            for pred in y_pred_gen:
                y_pred.append(pred['class_ids'][0])

            print(model_name + " => " + str(Deep_eval_result["accuracy"]) + " in " + str(
                time.process_time() - last_time) + " seconds\n")

            if Deep_eval_result["accuracy"] > best_acc:
                best_acc = Deep_eval_result["accuracy"]
                best_settings = model_name + "," + str(Deep_eval_result["accuracy"]) + "," + str(
                    time.process_time() - last_time)

            with open(dnn_log_csv_file, 'a') as log_file:
                log_file.write(model_name + "," + str(Deep_eval_result["accuracy"]) + "," + str(
                    time.process_time() - last_time) + "\n")
                log_file.close()
            with open(dnn_log_file, 'a') as log_file:
                log_file.write(model_name + "," + str(Deep_eval_result["accuracy"]) + "," + str(
                    time.process_time() - last_time) + "\n" + classification_report(y_test, y_pred))
                log_file.close()

print("Best Accuracy and Settings as Follows:")
print(best_settings)
print(str(best_acc))
