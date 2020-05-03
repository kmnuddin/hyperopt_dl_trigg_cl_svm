from hyperopt_svm import run_svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from hyperopt import Trials, fmin, tpe, hp

import numpy as np
import pandas as pd

import pickle
import os
import traceback

data = pd.read_csv("data_15vs3/data.csv", header=None).values
labels = np.ravel(pd.read_csv("data_15vs3/labels.csv", header=None).values)

scaler = MinMaxScaler()

data = scaler.fit_transform(data)

print('Data Transformed')

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=33)

space = {
    'data': {
        'type': '15vs3',

        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test
    },
    'params': {
        'C': hp.lognormal('C', 0,1),
        'kernel': hp.choice('kernel', ['rbf','poly', 'linear', 'sigmoid']),
        'degree': hp.choice('degree', range(1,15)),
        'gamma': hp.choice('gamma', ['scale', 'auto'])
    }
}

def run_trial():

    max_evals = nb_evals = 1

    try:
        trials = pickle.load(open("results_15vs3.pkl", "rb"))
        print("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
        print("Rerunning from {} trials to add another one.".format(
            len(trials.trials)))

    except:
        trials = Trials()
        print("Starting from scratch: new trials.")

    print("\nSTARTED OPTIMIZATION STEP.\n")
    best = fmin(run_svm, space, algo=tpe.suggest, trials=trials, max_evals=max_evals)

    pickle.dump(trials, open("results_15vs3.pkl", "wb"))

    print("\nOPTIMIZATION STEP COMPLETE.\n")



if __name__ == "__main__":

    while True:
        try:
            run_trial()
        except Exception as err:

            print(str(err))
            print(str(traceback.format_exc()))
            break;
