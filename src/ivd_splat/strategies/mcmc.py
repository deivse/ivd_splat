from dataclasses import dataclass
import typing

from gsplat.strategy import MCMCStrategy as GSplatMCMCStrategy

from ivd_splat.strategies.base import IVDSplatBaseStrategy


@dataclass
class MCMCStrategy(GSplatMCMCStrategy, IVDSplatBaseStrategy):
    CONFIG_SERIALIZATION_IGNORED_FIELDS: typing.ClassVar[set[str]] = {
        "verbose",
    }
