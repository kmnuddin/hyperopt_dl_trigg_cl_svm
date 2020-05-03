from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from bson import json_util
import json

class Helper:

    def __init__(self, result_dir):
        self.model_directory = 'models/'
        self.results_directory = result_dir

        if not os.path.exists(self.model_directory):
            os.mkdir(self.model_directory)

        if not os.path.exists(self.results_directory):
            os.mkdir(self.results_directory)



    def load_json_result(self, best_result_name):
        """Load json from a path (directory + filename)."""
        result_path = os.path.join(self.results_directory, best_result_name)
        with open(result_path, 'r') as f:
            return json.JSONDecoder().decode(
                f.read()
            )


    def load_best_hyperspace(self):
        results = [
            f for f in list(sorted(os.listdir(self.results_directory))) if 'json' in f
        ]
        if len(results) == 0:
            return None

        best_result_name = results[-1]
        return self.load_json_result(best_result_name)
