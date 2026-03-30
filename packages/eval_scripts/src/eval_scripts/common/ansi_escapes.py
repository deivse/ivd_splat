from enum import Enum
from typing import Self


class ANSIEscapes(Enum):
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    END_SEQUENCE = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    @staticmethod
    def by_name(name: str):
        return getattr(ANSIEscapes, name.upper())

    @staticmethod
    def format(text: str, escape: str | Self):
        seq = escape if isinstance(escape, ANSIEscapes) else ANSIEscapes.by_name(escape)
        return f"{seq.value}{text}{ANSIEscapes.END_SEQUENCE.value}"


def ansiesc_print(value: str, escape: str | ANSIEscapes):
    print(ANSIEscapes.format(value, escape))
