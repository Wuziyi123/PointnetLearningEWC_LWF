import torch
from torch import nn


class PointNet(nn.Module):

    def __init__(self, sampling, z_size, batch_norm):
        super().__init__()

        if batch_norm:
            self._conv1 = nn.Sequential(
                nn.Conv1d(3, 64, 1),
                nn.BatchNorm1d(64),
                nn.ReLU()
            )
        else:
            self._conv1 = nn.Sequential(
                nn.Conv1d(3, 64, 1),
                nn.ReLU()
            )

        if batch_norm:
            self._conv2 = nn.Sequential(
                nn.Conv1d(64, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(),

                nn.Conv1d(128, z_size, 1),
                nn.BatchNorm1d(z_size),
                nn.ReLU()
            )
        else:
            self._conv2 = nn.Sequential(
                nn.Conv1d(64, 128, 1),
                nn.ReLU(),

                nn.Conv1d(128, z_size, 1),
                nn.ReLU()
            )

        self._pool = nn.MaxPool1d(sampling)
        self._flatten = nn.Flatten(1)

    def forward(self, x):
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._pool(x)
        return self._flatten(x)


class PointNetClassifier(nn.Module):

    def __init__(self, sampling, z_size, batch_norm):
        super().__init__()
        self._pn = PointNet(sampling, z_size, batch_norm)
        self.sampling = sampling
        self._keys_active = ['t1', 't2', 't3', 't4', 't5']

        if batch_norm:
            self._dense = nn.Sequential(
                nn.Linear(z_size, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU()
            )
        else:
            self._dense = nn.Sequential(
                nn.Linear(z_size, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU()
            )

        self._final = nn.ModuleDict()

    def add_final_layer(self, key, num_classes, dropout=0.3):
        self._final[key] = nn.Sequential(
            nn.Linear(256, num_classes),
            nn.Dropout(dropout)
        )

    def set_active_keys(self, keys):
        self._keys_active = keys
        for key in self._final.keys():
            if key in self._keys_active:
                self._final[key].requires_grad = True
            else:
                self._final[key].requires_grad = False

    def forward(self, x):
        x = self._pn(x)
        x = self._dense(x)
        out = {}
        for key in self._keys_active:
            out[key] = self._final[key](x)
        # out = self._final(x)
        return out
