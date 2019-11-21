import csv
from KMeans.Model import *
import random
import copy
import math
from typing import Tuple


class Algorithm:
    def __init__(self, dataset_file, attrib_len, class_index, num_clusters: int, selected_attrib_idx: Tuple = None):
        self.selected_attrib_idx = selected_attrib_idx
        self.num_clusters = num_clusters
        self.dataset_file = dataset_file
        self.attrib_len = attrib_len
        self.class_index = class_index
        self.min_attribute = []
        self.max_attribute = []
        self.dataset = []
        self.centroids = []
        try:
            self.load_dataset()
        except Exception as e:
            print(e)

    def load_dataset(self):
        with open(self.dataset_file) as file_csv:
            rows = csv.reader(file_csv, delimiter=',')
            for idx, row in enumerate(rows):
                if len(row) < self.attrib_len:
                    continue
                temp = []
                class_name = None
                for idx_attrib, attrib_val in enumerate(row):
                    if idx_attrib == self.class_index:
                        class_name = attrib_val
                    else:
                        temp_val = attrib_val

                        try:
                            temp_val = float(attrib_val)
                        except:
                            pass

                        temp.append(temp_val)

                        if idx == 0:
                            self.min_attribute.append(temp_val)
                            self.max_attribute.append(temp_val)
                        else:
                            if temp_val < self.min_attribute[idx_attrib]:
                                self.min_attribute[idx_attrib] = temp_val
                            if temp_val > self.max_attribute[idx_attrib]:
                                self.max_attribute[idx_attrib] = temp_val

                self.dataset.append(Point(class_name, temp))

    def generate_random_centroids(self):
        self.centroids = []
        for i in range(self.num_clusters):
            temp_pos = []
            for i_attr in range(self.attrib_len):
                temp_pos.append(
                    random.random() * (self.max_attribute[i_attr] - self.min_attribute[i_attr]) + self.min_attribute[
                        i_attr])
            self.centroids.append(Centroid('c{}'.format(i), temp_pos, []))

    def calc_distance(self, posA: List[float], posB: List[float]):
        if len(posA) > len(posB):
            posA, posB = posB, posA
        val = 0
        for idx, x in enumerate(posA):
            if self.selected_attrib_idx is not None and idx not in self.selected_attrib_idx:
                continue
            val += math.pow(posA[idx] - posB[idx], 2)
        return math.sqrt(val)

    @staticmethod
    def is_same_position(old_centroids: List[Centroid], new_centroids: List[Centroid]):
        for idx, centroid in enumerate(old_centroids):
            for idx_centroid, pos_centroid in enumerate(centroid.pos):
                if old_centroids[idx].pos[idx_centroid] != new_centroids[idx].pos[idx_centroid]:
                    return False
        return True

    def find_nearest_centroid(self):
        clear_member = True
        for point in self.dataset:
            nearest_idx, nearest_val = None, 9999999999
            for idx, centroid in enumerate(self.centroids):
                if clear_member:
                    centroid.members = []
                temp_dist = self.calc_distance(centroid.pos, point.pos)
                if temp_dist < nearest_val:
                    nearest_idx, nearest_val = idx, temp_dist
            self.centroids[nearest_idx].members.append(point)
            clear_member = False

    def update_centroids(self):
        for centroid in self.centroids:
            if len(centroid.members) <= 0:
                continue
            first_member = True
            new_pos = []
            for point_member in centroid.members:
                for idx, point_pos in enumerate(point_member.pos):
                    if first_member:
                        new_pos.append(0)
                    new_pos[idx] += point_pos
                first_member = False

            for idx, value in enumerate(new_pos):
                new_pos[idx] /= len(centroid.members)
            centroid.pos = new_pos

    def print_centroids(self):
        for centroid in self.centroids:
            print(centroid.__dict__, )
        print()

    def train(self) -> List[Centroid]:
        self.generate_random_centroids()
        running = True
        # self.print_centroids()
        iteration = 0
        while running:
            iteration += 1
            old_centroids = copy.deepcopy(self.centroids)
            self.find_nearest_centroid()
            self.update_centroids()
            if Algorithm.is_same_position(old_centroids, self.centroids):
                running = False
        # self.print_centroids()
        print('Iteration\t', iteration)
        return self.centroids
