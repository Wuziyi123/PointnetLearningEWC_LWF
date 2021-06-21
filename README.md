# PointnetLearningEWC_LWF
Implementation of [Learning without Forgetting](https://arxiv.org/pdf/1606.09282.pdf) and [Elastic Weight Consolidation](https://arxiv.org/pdf/1612.00796.pdf) (both task-incremental) for PointNet classifier.

# Continual Learning
Continual learning is the extension of the original classifier in order to carry out subsequent tasks in such a way as to best preserve the ability to deal with the previous ones. There are three major scenarios:
* task-incremental
* domain-incremental
* class-incremental

To get more info check out [this](https://arxiv.org/pdf/1904.07734.pdf) paper.

# PointNet
PointNet is a neural network architecture designed to classify and segment point clouds. Point cloud is in fact a set of points. Due to the nature of the set as a structure (no order), transformations are required that render the model invariant with respect to the input (n! permutations, where n is a cardinality of a set).

# Usage

To train model use `train_ewc.py` or `train_lwf.py` with flags:
```
  -h, --help            show this help message and exit
  -e MAX_EPOCHS, --epochs MAX_EPOCHS
                        Maximal number of epochs to train network
  -bn BATCH_NORM, --batch_norm BATCH_NORM
                        Whether to use batch_norm or not
  -bs BATCH_SIZE, ---batch_size BATCH_SIZE
                        Batch size
  -lr LR, --learning_rate LR
                        Learning rate
  -p PATIENCE, --patience PATIENCE
                        How many epochs to tolerate no improvement
  -a ALPHA, --alpha ALPHA
                        Alpha parameter for regularization
  -z Z_SIZE, --z_size Z_SIZE
                        Size of final transformation vector
  -s SAMPLING, --sampling SAMPLING
                        Number of points in point cloud
  -v VERBOSE, --verbose VERBOSE
                        Whether to print information or log to file only
```

Experiments were made using [ModelNet10](https://modelnet.cs.princeton.edu/) dataset. The dataset was splitted into 5 tasks:
* t1 chair, sofa
* t2 bed, monitor
* t3 table, toilet
* t4 dresser, night_stand
* t5 desk, bathtub
