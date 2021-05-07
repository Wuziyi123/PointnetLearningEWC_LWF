import math
import numpy as np
from torchvision import transforms


class Normalize:

    def __call__(self, x):
        x -= np.mean(x, axis=0)
        x = x / np.max(np.linalg.norm(x, axis=1))
        return x


class RandomRotation:

    def __call__(self, x):
        phi = np.random.random(1) * 2 * np.pi
        rotation_matrix = np.array([
            [math.cos(phi), -math.sin(phi), 0],
            [math.sin(phi), math.cos(phi), 0],
            [0, 0, 1]
        ])
        return rotation_matrix.dot(x.T).T


class RandomNoise:
    def __call__(self, pointcloud):
        noise = np.random.normal(0, 0.02, pointcloud.shape)

        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud


def default_transforms():
    return transforms.Compose([
        Normalize(),
        RandomRotation(),
        RandomNoise(),
        transforms.ToTensor()
    ])


def test_transforms():
    return transforms.Compose([
        Normalize(),
        transforms.ToTensor()
    ])
