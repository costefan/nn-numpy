from utils.dataset_parser import parse_dataset
from nn.runner import Runner


if __name__ == '__main__':
    X, y = parse_dataset(dataset_name='spambase')

    Runner.run(X, y)
