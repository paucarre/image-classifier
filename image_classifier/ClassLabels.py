import json
import os
class ClassLabels():

    def __init__(self, dataset_folder):
        labels_path = os.path.join(dataset_folder, f"labels.json")
        with open(labels_path, "r") as labels_file:
            self.labels = json.load(labels_file)
