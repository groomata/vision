from hydra.core.config_store import ConfigStore

from groovis.configs import full_builds
from groovis.loss import SimCLRLoss

SimCLRLossConfig = full_builds(SimCLRLoss)


def _register_configs():
    cs = ConfigStore.instance()

    cs.store(
        group="loss",
        name="nt_xent_low",
        node=SimCLRLossConfig(temperature=0.05),
    )
    cs.store(
        group="loss",
        name="nt_xent_medium",
        node=SimCLRLossConfig(temperature=0.1),
    )
    cs.store(
        group="loss",
        name="nt_xent_high",
        node=SimCLRLossConfig(temperature=0.5),
    )
