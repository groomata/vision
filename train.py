import warnings

from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

import hydra

# from hydra.utils import instantiate
from rich import print

from groovis.schema import Cfg

# from groovis.train import train


@hydra.main(
    config_path="configs",
    config_name="default",
    version_base="1.2",
)
def main(config: Cfg):

    # architecture = instantiate(config.architecture)

    # print(architecture)

    print(config)
    # train(config=config)


if __name__ == "__main__":
    main()
