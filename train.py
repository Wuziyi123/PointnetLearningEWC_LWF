from core import train_model_lwf, parse_args

PATH = r'C:\Users\konra\Desktop\CloudOfPointsLWE\HModelNet10'
TASKS = {
    't1': ['chair', 'sofa'],
    't2': ['bed', 'monitor'],
    't3': ['table', 'toilet'],
    't4': ['dresser', 'night_stand'],
    't5': ['desk', 'bathtub']
}


def main():
    args = parse_args()
    train_model_lwf(
        PATH,
        TASKS,
        args.sampling,
        args.z_size,
        args.batch_norm,
        args.verbose,
        args.max_epochs,
        args.batch_size,
        args.lr,
        args.patience,
        args.alpha
    )


if __name__ == '__main__':
    main()
