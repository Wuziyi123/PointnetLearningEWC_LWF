import os
import pandas as pd


class RaportGenerator:


    def __init__(self, tasks, train_path):
        columns = list()
        self._idx = 0
        self._save_path = train_path
        for task in tasks:
            columns.append(f'{task}_train')
            columns.append(f'{task}_test')
        self._df = pd.DataFrame(columns=columns)

    def add_info(self, train_accuracy, test_accuracy):
        info = []
        for k1, k2 in zip(train_accuracy, test_accuracy):
            info.append(train_accuracy[k1])
            info.append(test_accuracy[k2])
        self._df.loc[self._idx] = info
        self._idx += 1

    def save(self):
        self._df.to_csv(os.path.join(self._save_path, 'raport.csv'))
