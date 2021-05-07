import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        default=5,
        help='Maximal number of epochs to train network',
        dest='max_epochs'
    )
    parser.add_argument(
        '-bn',
        '--batch_norm',
        type=bool,
        default=True,
        help='Whether to use batch_norm or not',
        dest='batch_norm'
    )
    parser.add_argument(
        '-bs',
        '---batch_size',
        type=int,
        default=10,
        help='Batch size',
        dest='batch_size'
    )
    parser.add_argument(
        '-lr',
        '--learning_rate',
        type=float,
        default=0.0001,
        help='Learning rate',
        dest='lr'
    )
    parser.add_argument(
        '-p',
        '--patience',
        type=int,
        default=7,
        help='How many epochs to tolerate no improvement',
        dest='patience'
    )
    parser.add_argument(
        '-a',
        '--alpha',
        type=float,
        default=2,
        help='Alpha parameter for regularization',
        dest='alpha'
    )
    parser.add_argument(
        '-z',
        '--z_size',
        type=int,
        default=1024,
        help='Size of final transformation vector',
        dest='z_size'
    )
    parser.add_argument(
        '-s',
        '--sampling',
        type=int,
        default=1024,
        help='Number of points in point cloud',
        dest='sampling'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        type=bool,
        default=True,
        help='Whether to print information or log to file only',
        dest='verbose'
    )
    return parser.parse_args()

