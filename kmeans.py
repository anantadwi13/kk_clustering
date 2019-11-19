import csv
import os


class KMeans:
    def __init__(self, dataset_file, attrib_len, class_index):
        self.dataset_file = dataset_file
        self.attrib_len = attrib_len
        self.class_index = class_index
        self.dataset = []
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
                self.dataset.append({'attributes': temp, 'class': class_name})
            print(self.dataset)


kmeans = KMeans(os.path.dirname(os.path.abspath(__file__)) + '\\dataset\\iris.data', 4, 4)
