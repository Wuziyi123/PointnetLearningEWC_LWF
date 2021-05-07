import os
import glob
import torch
import trimesh
import logging

logger = logging.getLogger('trimesh')
logger.setLevel(logging.WARNING)
from torch.utils.data import Dataset
from core.dataset.transforms import default_transforms


class Data(Dataset):

    def __init__(self, path, classes, sampling, kind='train', transforms=default_transforms()):
        self._path = path
        self._sampling = sampling
        self._kind = kind
        self._classes = self._setup_classes(classes)
        self._data = self._setup_data(classes)
        self._transforms = transforms

    def _setup_classes(self, classes):
        classes_map = dict()
        for i, c in enumerate(classes):
            classes_map[c] = i
        return classes_map

    def _setup_data(self, classes):
        paths = list()
        for c in classes:
            paths.extend(
                [(path, c) for path in glob.glob(os.path.join(self._path, c, self._kind, '*.off'))]
            )
        return paths

    def _get_item(self, idx):
        path, c = self._data[idx]
        x = self._apply_transform(path)
        return x.squeeze(0).float(), self._classes[c]

    def _apply_transform(self, path):
        mesh = trimesh.load(path).sample(self._sampling)
        return self._transforms(mesh)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._get_item(idx)


class LWFDataset(Dataset):

    def __init__(self, dataset: Data, model, prev_tasks, new_task):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cpu = torch.device('cpu')
        self._base_dataset = dataset
        self._new_task = new_task
        self._prev_tasks_output = []
        model.set_active_keys(prev_tasks)
        model.eval()
        with torch.no_grad():
            for i in range(len(dataset)):
                output = dict()
                x, _ = dataset[i]
                pred = model(x.unsqueeze(0).transpose(1, 2).to(device))
                for key in prev_tasks:
                    output[key] = pred[key].squeeze(0).to(cpu)
                self._prev_tasks_output.append(output)

    def __len__(self):
        return len(self._base_dataset)

    def __getitem__(self, idx):
        item = self._base_dataset[idx]
        return item, self._prev_tasks_output[idx]
