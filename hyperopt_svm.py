from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.svm import SVC

from bson import json_util
import json
import os
import pickle
from hyperopt import STATUS_OK

RESULTS_DIR = "results/"

def run_svm(args):

    data_type = args['data']['type']

    x_train = args['data']['x_train']
    x_test = args['data']['x_test']
    y_train = args['data']['y_train']
    y_test = args['data']['y_test']

    C = args['params']['C']
    kernel = args['params']['kernel']
    degree = args['params']['degree']
    gamma = args['params']['gamma']

    svc = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)

    svc.fit(x_train, y_train)

    y_pred = svc.predict(x_test)

    acc = accuracy_score(y_pred, y_test)
    mse = mean_squared_error(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    model_name = 'svm_{}_acc{}_{}'.format(data_type, str(acc), str(data_type))

    results = {
        'loss': -acc,

        'C': C,
        'kernel': kernel,
        'degree': degree,
        'gamma': gamma,

        'cr': cr,
        'acc': acc,
        'mse': mse,
        'status': STATUS_OK
    }
    print_json(results)
    save_json_result(model_name, results)
    pickle.dump(svc, open('models/{}.pkl'.format(model_name), 'wb'))

    return results


def save_json_result(model_name, result):
    """Save json to a directory and a filename."""
    result_name = '{}.txt.json'.format(model_name)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    with open(os.path.join(RESULTS_DIR, result_name), 'w') as f:
        json.dump(
            result, f,
            default=json_util.default, sort_keys=True,
            indent=4, separators=(',', ': ')
        )

def print_json(result):
    """Pretty-print a jsonable structure (e.g.: result)."""
    print(json.dumps(
        result,
        default=json_util.default, sort_keys=True,
        indent=4, separators=(',', ': ')
    ))
