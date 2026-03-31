import itertools
from pathlib import Path
from typing import Self
import typing


ParsedConfigStr = list[tuple[str, list[str]]]
"""A parsed config string is a list of (param_name, [possible_values]) tuples."""


class ParamList:
    """A ParamList is a list of parameters for a single invocation - tuple of (param_name, value) tuples."""

    def __init__(self, values: typing.Iterable[tuple[str, str]] | None = None):
        self.__values = tuple(values) if values is not None else ()

    def __iter__(self):
        return iter(self.__values)

    def validate(self, forbidden_param_names: set[str]):
        if self.__values is None:
            return
        for name, _ in self.__values:
            if (
                name in forbidden_param_names
                or name.replace("-", "_") in forbidden_param_names
            ):
                raise ValueError(
                    f"Parameter {name} can not be part of config string"
                    " (it's either set by evaluator automatically, or evaluator argument should be used instead of passing directly)"
                )

    def __str__(self):
        return str(self.__values)

    def with_prepended_params(
        self, new_params: Self | tuple[tuple[str, str]]
    ) -> "ParamList":
        return ParamList((*new_params, *(self.__values or ())))

    def make_config_name(
        self,
        renames: dict[str, str | None],
        extra_tags: list[str] | None = None,
    ) -> str:
        """
        Create a config name suitable for use in directory names from a ParamList.
        Args:
            renames: A mapping from parameter names to their desired names in the config string.
                If a parameter name maps to None, it will be omitted from the config string.
        """
        if self.__values is None or len(self.__values) == 0:
            return "default"

        out = []

        def with_tildes(n):
            return n.replace("_", "-")

        name: str | None
        for name, value in self.__values or ():
            if name is None:
                out.append(value)
                continue

            if with_tildes(name) in renames:
                name = renames[with_tildes(name)]

            out.append(f"{name}={value}")

        out.extend(extra_tags or [])

        if len(out) == 0:
            return "default"

        out = [x.replace("/", "_") for x in out]

        return "_".join(out)

    def __lt__(self, other: "ParamList") -> bool:
        return self.__values < other.__values


def load_configs(
    config_strings: list[str], configs_file: str | None
) -> list[ParamList]:
    """
    Load config strings from either a list of strings or a file, and parse them into ParamLists.
    Args:
        config_strings: List of config strings from CLI, or an empty list
        configs_file: Path to a file containing config strings from cli, one per line, or None
    Raises:
        ValueError: If both or neither of config_str and configs_file are specified.
    Returns:
        List of ParamLists - each representing a single set of parameters with which to run the method.
    """
    param_lists: list[ParamList] = []
    for config_str in _load_configs_from_args(config_strings, configs_file):
        param_lists.extend(_parse_config_string(config_str))
    return list(set(param_lists))  # deduplicate


def _load_configs_from_args(configs: list[str], configs_file: str | None) -> list[str]:
    """
    Get the list of config strings from either a list of strings or a file.
    Args:
        configs: List of config strings.
        configs_file: Path to a file containing config strings, one per line.
    Raises:
        ValueError: If both or neither of configs and configs_file are specified.
    Returns:
        List of config strings.
    """
    num_exclusive_options_specified = int(
        (len(configs) > 0) and not configs == [""]
    ) + int(configs_file is not None)

    if num_exclusive_options_specified > 1:
        raise ValueError("Only one of  {--configs, --configs-file} may be specified.")

    if configs_file is None:
        return configs

    with Path(configs_file).open("r", encoding="utf-8") as file:
        return [
            line.strip()
            for line in file.readlines()
            if len(line.strip()) != 0 and not line.strip().startswith("#")
        ]


def _parse_config_string(config_str: str) -> list[ParamList]:
    """
    Parse a config string into a list of ParamLists, one for each combination of parameters specified by the config string.
    # Basic example
    --alignment-method={ransac,msac}
    """

    if config_str.strip() == "<default>" or config_str.strip() == "":
        return [ParamList()]

    parts: list[str] = []
    current_part = ""
    brace_count = 0
    in_quotes = False
    quote_char = None

    for char in config_str:
        if char in ('"', "'") and not in_quotes:
            in_quotes = True
            quote_char = char
            current_part += char
        elif char == quote_char and in_quotes:
            in_quotes = False
            quote_char = None
            current_part += char
        elif char == "{" and not in_quotes:
            brace_count += 1
            current_part += char
        elif char == "}" and not in_quotes:
            brace_count -= 1
            current_part += char
        elif char == " " and brace_count == 0 and not in_quotes:
            if current_part:
                parts.append(current_part)
                current_part = ""
        else:
            current_part += char

    if current_part:
        parts.append(current_part)

    parsed: ParsedConfigStr = []
    for part in parts:
        eq_pos = part.find("=")
        if eq_pos == -1:
            raise ValueError(
                f"'=' not found in \"{part}\" All config string param definitions must be formatted as"
                + "key={value1, value2, ...}"
            )
        name = part[:eq_pos].removeprefix("-").removeprefix("-")
        name = name.replace("-", "_")

        if part[eq_pos + 1] == "{":  # List of options
            if not part[-1] == "}":
                raise ValueError("Invalid config string: unclosed {} at " + part)
            values = part[eq_pos + 2 : -1].replace(" ", "").split(",")
            parsed.append((name, values))
            continue

        if "{" in part or "}" in part:
            raise ValueError(
                "{} contained in part, but open brace is not on first pos: " + part
            )
        value = part[eq_pos + 1 :]
        parsed.append((name, [value]))

    possible_vals_with_names: list[list[tuple[str, str]]] = []
    for name, values in parsed:
        possible_vals_with_names.append([(name, val) for val in values])

    all_combinations = {
        ParamList(x) for x in itertools.product(*possible_vals_with_names)
    }
    # Pass through set to deduplicate
    return list(set(all_combinations))
