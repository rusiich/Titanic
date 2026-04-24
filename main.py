from configs import config
from utils import set_seed

def fit(config):
    set_seed(config.general.seed)


if __name__ == '__main__':
    fit(config)