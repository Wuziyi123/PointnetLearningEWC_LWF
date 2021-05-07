import os
from datetime import datetime


DATA = datetime.now().strftime("%d.%m.%Y-%H.%M.%S")

dirs = ['checkpoints', 'logs', 'results']


if not os.path.exists('./training'):
    os.mkdir('./training')

train_path = os.path.join('.', 'training', DATA)
os.mkdir(train_path)
for dir_ in dirs:
    os.mkdir(os.path.join(train_path, dir_))


from core.early_stopping import EarlyStopper
from core.lwf_loss import KnowledgeDistillationLoss
from core.model import PointNetClassifier
from core.parse_args import parse_args
from core.train_lwf_model import train_model_lwf
