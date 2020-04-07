from hyperopt_svm import run_svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from hyperopt import Trials, fmin, tpe, hp

import pickle
import os
import traceback


def run_trial():

    max_evals = nb_evals = 1

    try:
        trials = pickle.load(open("results.pkl", "rb"))
        print("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
        print("Rerunning from {} trials to add another one.".format(
            len(trials.trials)))

    except:
        trials = Trials()
        print("Starting from scratch: new trials.")


    best = fmin(run_svm, space, algo=tpe.suggest, trials=trials, max_eval=max_evals)

    pickle.dump(trials, open("results.pkl", "rb"))

    print("\nOPTIMIZATION STEP COMPLETE.\n")



if __name__ == "__main__":

    while True:
        try:
            run_trial()
        except Exception as err:

            print(str(err))
            print(str(traceback.format_exc()))
