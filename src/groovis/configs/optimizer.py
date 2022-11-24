import torch_optimizer
from hydra.core.config_store import ConfigStore
from torch import optim

from groovis.configs import partial_builds

SGDConfig = partial_builds(optim.SGD, lr=0.005)
RMSPropConfig = partial_builds(optim.RMSprop, lr=0.005)
AdamConfig = partial_builds(optim.Adam, lr=0.005)
AdamWConfig = partial_builds(optim.AdamW, lr=0.005)
RAdamConfig = partial_builds(optim.RAdam, lr=0.005)

LARSConfig = partial_builds(torch_optimizer.LARS, lr=0.005)


def _register_configs():
    cs = ConfigStore.instance()

    cs.store(group="optimizer", name="sgd", node=SGDConfig)
    cs.store(group="optimizer", name="rmsprop", node=RMSPropConfig)
    cs.store(group="optimizer", name="adam", node=AdamConfig)
    cs.store(group="optimizer", name="adamw", node=AdamWConfig)
    cs.store(group="optimizer", name="radam", node=RAdamConfig)

    cs.store(group="optimizer", name="lars", node=LARSConfig)
