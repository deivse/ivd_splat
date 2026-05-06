import importlib
import os
from pathlib import Path
import sys
import logging

from ivd_splat.strategies.base import IVDSplatBaseStrategy
from ivd_splat.strategies.inria import INRIAStrategy
from ivd_splat.strategies.revdgs import RevDGSStrategy

from .default_without_adc import DefaultWithoutADCStrategy
from .default_with_gaussian_cap import DefaultWithGaussianCapStrategy
from .idhfr import IDHFRStrategy
from .mcmc import MCMCStrategy

_LOGGER = logging.getLogger(__name__)


def discover_additional_strategy_classes():
    strategies_dir = os.getenv("IVD_SPLAT_ADDITIONAL_STRATEGIES_DIR")
    if strategies_dir is None:
        return []
    sys.path.append(strategies_dir)
    strategy_classes = []

    for file in Path(strategies_dir).iterdir():
        if not file.suffix == ".py" or file.name.startswith("__"):
            continue
        module_name = file.stem
        module = importlib.import_module(module_name)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, IVDSplatBaseStrategy)
                and attr is not IVDSplatBaseStrategy
            ):
                strategy_classes.append(attr)
                # from module_name import attr_name
                setattr(sys.modules[__name__], attr_name, attr)

    _LOGGER.info(
        f"Discovered additional strategy classes: {[cls.__name__ for cls in strategy_classes]}"
    )
    return strategy_classes


ADDITIONAL_STRATEGY_CLASSES = discover_additional_strategy_classes()
ALL_STRATEGY_CLASSES = [
    DefaultWithoutADCStrategy,
    DefaultWithGaussianCapStrategy,
    MCMCStrategy,
    IDHFRStrategy,
    INRIAStrategy,
    RevDGSStrategy,
    *ADDITIONAL_STRATEGY_CLASSES,
]

__all__ = [
    "DefaultWithoutADCStrategy",
    "DefaultWithGaussianCapStrategy",
    "MCMCStrategy",
    "IDHFRStrategy",
    "INRIAStrategy",
    "RevDGSStrategy",
] + [cls.__name__ for cls in ADDITIONAL_STRATEGY_CLASSES]
