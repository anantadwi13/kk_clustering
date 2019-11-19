import csv
import os
from typing import List
import math
import copy


class Data:
    def __init__(self, class_name: str, attributes: List):
        self.class_name = class_name
        self.attributes = attributes


class ClusterWeight:
    def __init__(self, name: str, weights: list):
        self.name = name
        self.weights = weights


class SOM:
    STATUS_ERROR = -1
    STATUS_STOPPED = 0
    STATUS_PREPARING = 1
    STATUS_RUNNING = 2
    END_LEARNING_RATE = 0.01

    def __init__(self, dataset_file, attrib_len, class_index, clusterweights: List[ClusterWeight], learningrate: float,
                 radius: float):
        self.dataset_file = dataset_file
        self.attrib_len = attrib_len
        self.class_index = class_index
        self.clusterweights = clusterweights
        self.learningrate = learningrate
        self.radius = radius
        self.dataset: List[Data] = []
        self.status = SOM.STATUS_PREPARING

        for cluster in self.clusterweights:
            if len(cluster.weights) != self.attrib_len:
                self.status = SOM.STATUS_ERROR
                print("Jumlah attribute weight tidak sama dengan jumlah attribute dataset")
                return
        try:
            self.load_dataset()
        except Exception as e:
            print(e)

    def load_dataset(self):
        with open(self.dataset_file) as file_csv:
            rows = csv.reader(file_csv, delimiter=',')
            for row in rows:
                if len(row) < self.attrib_len:
                    continue
                temp = []
                class_name = None
                for idx, attrib_val in enumerate(row):
                    if idx == self.class_index:
                        class_name = attrib_val
                    else:
                        try:
                            temp.append(float(attrib_val))
                        except:
                            temp.append(attrib_val)
                self.dataset.append(Data(class_name, temp))

    def calculate_distance(data: Data, cluster_weights: List[ClusterWeight]) -> dict:
        min_distance = 100000000
        min_idx = None
        for idx_cluster, cluster_weight in enumerate(cluster_weights):
            temp = 0
            for idx, weight in enumerate(cluster_weight.weights):
                temp += math.pow(data.attributes[idx] - weight, 2)
            if temp < min_distance:
                min_idx = idx_cluster
                min_distance = temp
        return {'index': min_idx, 'distance': min_distance}

    def update_weight(input: Data, clusterweights: List[ClusterWeight], temp_distance, learningrate):
        for idx, cluster_weight in enumerate(clusterweights):
            if temp_distance['index'] == idx:
                temp_weight = []
                for idx_weight, weight in enumerate(cluster_weight.weights):
                    temp_weight.append(((1-learningrate) * weight) + (learningrate * input.attributes[idx_weight]))
                cluster_weight.weights = temp_weight

    def print_weight(clusterweights: List[ClusterWeight]):
        for cluster_weight in clusterweights:
            print(cluster_weight.__dict__)

    def train(self):
        learningrate = copy.deepcopy(self.learningrate)
        clusterweights = copy.deepcopy(self.clusterweights)
        while learningrate > SOM.END_LEARNING_RATE:
            for row in self.dataset:
                SOM.print_weight(clusterweights)
                temp_distance = SOM.calculate_distance(row, clusterweights)
                SOM.update_weight(row, clusterweights, temp_distance, learningrate)
            learningrate *= 0.5


if __name__ == '__main__':
    som = SOM(os.path.dirname(os.path.abspath(__file__)) + '\\dataset\\coba.data', 3, -1,
              [ClusterWeight('c1', [0.5, 0.6, 0.8]), ClusterWeight('c2', [0.4, 0.2, 0.5])], 0.5, 0)
    som.train()
    print('========================================')
    som.train()
