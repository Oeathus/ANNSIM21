from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow import keras
import pandas as pd
import numpy as np
import os
import time

data_folder = "data" + os.path.sep + "classifier_data" + os.path.sep
lstmModelPath = "models" + os.path.sep + "classifier" + os.path.sep + "lstm" + os.path.sep

datasets = os.listdir(data_folder)

lstm_log_csv_file = "logs" + os.path.sep + "classifier" + os.path.sep + "lstm.log.csv"
lstm_log_file = "logs" + os.path.sep + "classifier" + os.path.sep + "lstm.log"

sequence_lengths = [3, 7]
learning_rate = 0.001
epochs = 10
best_acc = 0
best_settings = ""

if not os.path.exists(lstm_log_csv_file):
    log_file = open(lstm_log_csv_file, 'w')
    log_file.close()

if not os.path.exists(lstm_log_file):
    log_file = open(lstm_log_file, 'w')
    log_file.close()

for dataset in datasets:
    dataset_name = os.path.splitext(dataset)[0]
    dataset_name = dataset_name.replace('.', ',')
    for sequence_length in sequence_lengths:
        last_time = time.process_time()
        name = dataset_name + "," + str(sequence_length)

        X = pd.read_csv(data_folder + dataset)
        X.drop(['interval_id'], axis=1, inplace=True)
        y = X.pop('rains_next_interval')
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)

        dataset_train = keras.preprocessing.timeseries_dataset_from_array(
            X_train,
            y_train,
            sequence_length=sequence_length,
            sampling_rate=1,
            batch_size=256,
        )
        dataset_val = keras.preprocessing.timeseries_dataset_from_array(
            X_test,
            y_test,
            sequence_length=sequence_length,
            sampling_rate=1,
            batch_size=256,
        )

        for batch in dataset_train.take(1):
            inputs, targets = batch

        model = keras.models.Sequential()
        model.add(keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2])))
        model.add(keras.layers.LSTM(32))
        model.add(keras.layers.Dense(1))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      loss="mse", metrics=['accuracy'])
        path_checkpoint = lstmModelPath + name + ".h5"
        es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

        modelckpt_callback = keras.callbacks.ModelCheckpoint(
            monitor="val_loss",
            filepath=path_checkpoint,
            verbose=1,
            save_weights_only=True,
            save_best_only=True,
        )
        if not os.path.exists(path_checkpoint):
            history = model.fit(
                dataset_train,
                epochs=epochs,
                validation_data=dataset_val,
                callbacks=[es_callback, modelckpt_callback],
                verbose=0
            )
        else:
            model.load_weights(path_checkpoint)

        y_pred = model.predict_classes(dataset_val, verbose=0)
        y_pred = y_pred.flatten()
        y_test = y_test[sequence_length - 1:]

        acc_oldfashion = (y_test.values == y_pred) * 1
        acc = np.mean(acc_oldfashion)

        if acc > best_acc:
            best_acc = acc
            best_settings = name

        print(name + "," + str(acc))
        with open(lstm_log_csv_file, 'a') as log_file:
            log_file.write(name + "," + str(acc) + "\n")
            log_file.close()
        with open(lstm_log_file, 'a') as log_file:
            log_file.write(name + "," + str(acc) + "\n")
            log_file.write(classification_report(y_test, y_pred) + "\n\n")
            log_file.close()

print("Best Accuracy and Settings as Follows:")
print(best_settings)
print(str(best_acc))
