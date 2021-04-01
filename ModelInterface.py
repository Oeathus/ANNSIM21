#!C:\Users\timha\Anaconda3\envs\pythonProject\python.exe
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, GRU, Bidirectional
from scipy.stats import pearsonr
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import os
import pickle


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


def wx_input_fn(features, targets=None, num_epochs=None, shuffle=True, batch_size=128):
    return tf.compat.v1.estimator.inputs.pandas_input_fn(x=features, y=targets,
                                                         num_epochs=num_epochs,
                                                         shuffle=shuffle,
                                                         batch_size=batch_size)


if __name__ == "__main__":
    model_type = 'classifier'
    model_source = 'ROC'
    model_normalization = 'None'
    model_method = 'knn'
    model_random_state = 'None'
    model_parameter = '8'
    dataset = 'ROC.None.csv'

    if '--help' in sys.argv:
        print("--model-type [classifier* | regressor]")
        print("--model-source [ROC* | ROC+BUF+SYR+ALB]")
        print("--model-normalization [None* | MinMax | ZScore]")
        print("--model-method [knn* | svm | rcc | dnn | dwnn | wnn | lstm | bidirectional | gru]")
        print("--model-random-state [None* | 0 | 42]")
        print("--model-parameter [various int]")
        print("    this parameter is specific to models that need it, search the model folders to see possible options")
        print("--dataset [filename]")
        print()
        print("Parameters with '*' are the defaults chosed when these flags are not given")
        print("Some combinations don't exist such as '--model-type classifier --model-method gru'")
        print("These options are used in combination to pick a dataset and model from their respective directories")
        print("Ex. '--model-type regressor")
        print("     --model-method knn")
        print("     --dataset ROC.MinMax.csv'")
        print("     --model-random-state None")
        print("     --model-parameter 11")
        print("     will get the dataset from ./data/regressor_data/ROC.MinMax.csv")
        print("     and will use the model from ./models/regressor/knn/ROC,MinMax,None,11")
        print("Ex. '--model-type classifier")
        print("     --model-method knn")
        print("     --dataset ROC.MinMax.csv'")
        print("     --model-random-state None")
        print("     --model-parameter 44")
        print("     will get the dataset from ./data/classifier_data/ROC.MinMax.csv")
        print("     and will use the model from ./models/regressor/knn/ROC,MinMax,None,44")
        exit(0)
    else:
        print("Use --help to see options")
    if '--model-type' in sys.argv:
        model_type = sys.argv[sys.argv.index('--model-type') + 1]
        print(f'Model type "{model_type}"')
    else:
        print(f'Model type not specified, using "{model_type}" as default')
    if '--model-source' in sys.argv:
        model_source = sys.argv[sys.argv.index('--model-source') + 1]
        print(f'Model source "{model_source}"')
    else:
        print(f'Model source not specified, using "{model_source}" as default')
    if '--model-normalization' in sys.argv:
        model_normalization = sys.argv[sys.argv.index('--model-normalization') + 1]
        print(f'Model normalization "{model_normalization}"')
    else:
        print(f'Model normalization not specified, using "{model_normalization}" as default')
    if '--model-method' in sys.argv:
        model_method = sys.argv[sys.argv.index('--model-method') + 1]
        print(f'Model method "{model_method}"')
    else:
        print(f'Model method not specified, using "{model_method}" as default')
    if '--model-random-state' in sys.argv:
        model_random_state = sys.argv[sys.argv.index('--model-random-state') + 1]
        print(f'Model random state "{model_random_state}"')
    else:
        print(f'Model random state not specified, using "{model_random_state}" as default')
    if '--model-parameter' in sys.argv:
        model_parameter = sys.argv[sys.argv.index('--model-parameter') + 1]
        print(f'Model parameter "{model_parameter}"')
    else:
        print(f'Model parameter state not specified, using "{model_parameter}" as default')
    if '--dataset' in sys.argv:
        dataset = sys.argv[sys.argv.index('--dataset') + 1]
        print(f'Dataset "{dataset}"')
    else:
        print(f'Dataset not specified, using "{dataset}" as default')

    dataset_path = 'data' + os.path.sep + model_type + '_data' + os.path.sep
    if not os.path.exists(dataset_path + dataset):
        print(f'Dataset "{dataset_path + dataset}" does not exist')
        exit(1)
    else:
        X = pd.read_csv(dataset_path + dataset)
        X.drop(['interval_id'], axis=1, inplace=True)

    if model_type == 'classifier':
        y = X.pop('rains_next_interval')
    elif model_type == 'regressor':
        y = X.pop('next_p01i')
    else:
        print(f'Model type "{model_type}" was not used, no models exist')
        exit(1)

    if model_method in ['lstm', 'bidirectional', 'gru']:
        model_name = model_source + ',' + model_normalization + ',' + model_parameter
    elif model_method in ['wnn', 'linear']:
        model_name = model_source + ',' + model_normalization + ',' + model_random_state
    else:
        model_name = model_source + ',' + model_normalization + ',' + model_random_state + ',' + model_parameter

    model_path = 'models' + os.path.sep + model_type + os.path.sep + model_method + os.path.sep
    if not os.path.exists(model_path + model_name):
        print(f'Model "{model_path + model_name}" does not exist')
        exit(1)
    else:
        if model_method in ['knn', 'svm', 'rcc', 'linear']:
            model = pickle.load(open(model_path + model_name, 'rb'))
            if model_method == 'rcc':
                y_pred = model.rc_classification(X.values)
                y_temp = pd.get_dummies(y)
                acc = (y_temp.values == y_pred) * 1
                accuracy = np.mean(acc, axis=0)[0]
                print(model_name + " => " + str(accuracy))
                print(classification_report(y_temp, y_pred))
            else:
                y_pred = model.predict(X)
                if model_type == 'classifier':
                    acc = (y == y_pred) * 1
                    accuracy = np.mean(acc, axis=0)
                    print(model_name + " => " + str(accuracy))
                    print(classification_report(y, y_pred))
                elif model_type == 'regressor':
                    r_squared = r2_score(y, y_pred)  # Best is 1.0
                    skMSE = mean_squared_error(y, y_pred, squared=True)  # Best is 0.0
                    skRMSE = mean_squared_error(y, y_pred, squared=False)  # Best is 0.0
                    sciPearson = pearsonr(y, y_pred)[0]  # Best is 1.0
                    print(model_name)
                    print(f'R^2: {r_squared}')
                    print(f'MSE: {skMSE}')
                    print(f'RMSE: {skRMSE}')
                    print(f'Pearson Correlation: {sciPearson}')
        elif model_type == 'classifier':
            if model_method in ['dnn', 'wnn', 'dwnn']:
                feature_columns = []
                for key in X.keys():
                    feature_columns.append(tf.feature_column.numeric_column(key=key))
                classes = len(set(y.values))
                inner_layers = [X.shape[1]] * int(model_parameter)

                if model_method == 'dnn':
                    model = tf.estimator.DNNClassifier(
                        feature_columns=feature_columns,
                        hidden_units=inner_layers,
                        n_classes=classes,
                        model_dir=model_path + model_name
                    )
                elif model_method == 'wnn':
                    model = tf.estimator.LinearClassifier(
                        feature_columns=feature_columns,
                        n_classes=classes,
                        model_dir=model_path + model_name
                    )
                elif model_method == 'dwnn':
                    model = tf.estimator.DNNLinearCombinedClassifier(
                        linear_feature_columns=feature_columns,
                        dnn_feature_columns=feature_columns,
                        dnn_hidden_units=inner_layers,
                        n_classes=classes,
                        model_dir=model_path + model_name)

                pred_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(X, batch_size=256, shuffle=False)
                y_pred_gen = model.predict(input_fn=pred_input_fn)
                y_pred = []
                for pred in y_pred_gen:
                    y_pred.append(pred['class_ids'][0])
                acc = (y == y_pred) * 1
                accuracy = np.mean(acc, axis=0)
                print(model_name + " => " + str(accuracy))
                print(classification_report(y, y_pred))
            elif model_method == 'lstm':
                dataset_train = keras.preprocessing.timeseries_dataset_from_array(
                    X, y,
                    sequence_length=int(model_parameter),
                    sampling_rate=1,
                    batch_size=256,
                )
                dataset_val = keras.preprocessing.timeseries_dataset_from_array(
                    X, y,
                    sequence_length=int(model_parameter),
                    sampling_rate=1,
                    batch_size=256,
                )
                for batch in dataset_train.take(1):
                    inputs, targets = batch
                model = keras.models.Sequential()
                model.add(Input(shape=(inputs.shape[1], inputs.shape[2])))
                model.add(LSTM(32))
                model.add(Dense(1))
                model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                              loss="mse", metrics=['accuracy'])
                model.load_weights(model_path + model_name)
                y_pred = model.predict_classes(dataset_val, verbose=0)
                y_pred = y_pred.flatten()
                y = y[int(model_parameter) - 1:]

                acc_oldfashion = (y.values == y_pred) * 1
                acc = np.mean(acc_oldfashion)
                print(model_name + " => " + str(acc))
                print(classification_report(y, y_pred))
        elif model_type == 'regressor':
            if model_method in ['dnn', 'wnn', 'dwnn']:
                feature_columns = [tf.feature_column.numeric_column(col) for col in X.columns]
                inner_layers = [X.shape[1]] * int(model_parameter)

                if model_method == 'dnn':
                    model = tf.estimator.DNNRegressor(
                        feature_columns=feature_columns,
                        hidden_units=inner_layers,
                        model_dir=model_path + model_name
                    )
                elif model_method == 'wnn':
                    model = tf.estimator.LinearRegressor(
                        feature_columns=feature_columns,
                        model_dir=model_path + model_name
                    )
                elif model_method == 'dwnn':
                    model = tf.estimator.DNNLinearCombinedRegressor(
                        linear_feature_columns=feature_columns,
                        dnn_feature_columns=feature_columns,
                        dnn_hidden_units=inner_layers,
                        model_dir=model_path + model_name)

                y_pred_gen = model.predict(input_fn=wx_input_fn(X, shuffle=False))
                i = 0
                y_pred = []
                for pred in y_pred_gen:
                    y_pred.append(pred["predictions"][0])
                    i += 1
                    if i >= X.shape[0]:
                        break
                r_squared = r2_score(y, y_pred)  # Best is 1.0
                skMSE = mean_squared_error(y, y_pred, squared=True)  # Best is 0.0
                skRMSE = mean_squared_error(y, y_pred, squared=False)  # Best is 0.0
                sciPearson = pearsonr(y, y_pred)[0]  # Best is 1.0
                print(model_name)
                print(f'R^2: {r_squared}')
                print(f'MSE: {skMSE}')
                print(f'RMSE: {skRMSE}')
                print(f'Pearson Correlation: {sciPearson}')
            elif model_method in ['lstm', 'bidirectional', 'gru']:
                X_series, y_series = create_dataset(X.values, y.values, int(model_parameter))

                if model_method == 'lstm':
                    model = create_model(64, LSTM, X_series.shape[1], X_series.shape[2])
                elif model_method == 'bidirectional':
                    model = create_model(64, Bidirectional, X_series.shape[1], X_series.shape[2])
                elif model_method == 'gru':
                    model = create_model(64, GRU, X_series.shape[1], X_series.shape[2])

                model.load_weights(model_path + model_name)
                y_pred = model.predict(X_series).flatten()
                r_squared = r2_score(y_series, y_pred)  # Best is 1.0
                skMSE = mean_squared_error(y_series, y_pred, squared=True)  # Best is 0.0
                skRMSE = mean_squared_error(y_series, y_pred, squared=False)  # Best is 0.0
                sciPearson = pearsonr(y_series, y_pred)[0]  # Best is 1.0
                print(model_name)
                print(f'R^2: {r_squared}')
                print(f'MSE: {skMSE}')
                print(f'RMSE: {skRMSE}')
                print(f'Pearson Correlation: {sciPearson}')
