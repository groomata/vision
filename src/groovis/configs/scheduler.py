from hydra.core.config_store import ConfigStore
from torch.optim.lr_scheduler import OneCycleLR

from groovis.configs import partial_builds

OneCycleLRConfig = partial_builds(
    OneCycleLR,
    max_lr="${optimizer.lr}",
    pct_start=0.2,
    anneal_strategy="linear",
    div_factor=1e3,
    final_div_factor=1e6,
)


def _register_configs():
    cs = ConfigStore.instance()

    cs.store(group="scheduler", name="onecycle", node=OneCycleLRConfig)
