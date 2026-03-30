import re

from tabulate import tabulate


def format_config_name_for_print(preset_name: str) -> str:
    return preset_name


class PresetFilter:
    def __init__(self, in_regex: str | None, out_regex: str | None):
        self.in_regex = re.compile(in_regex) if in_regex else None
        self.out_regex = re.compile(out_regex) if out_regex else None

    def allows(self, preset_name: str) -> bool:
        if self.in_regex and not self.in_regex.search(preset_name):
            return False
        if self.out_regex and self.out_regex.search(preset_name):
            return False
        return True


def preset_without_predictor(preset_id: str):
    KNOWN_PREDICTOR_IDS = [
        "metric3d",
        "unidepth",
        "depth_anything_v2_indoor",
        "depth_anything_v2_outdoor",
        "moge",
    ]
    for predictor_id in KNOWN_PREDICTOR_IDS:
        if predictor_id in preset_id:
            return preset_id.replace(f"{predictor_id}_", "")
    return preset_id


def format_best(val, output_fmt):
    MARKDOWN_FORMATS = ["github", "grid", "pipe", "jira", "presto", "pretty", "rst"]
    if output_fmt in MARKDOWN_FORMATS:
        return f"***{val}***"
    if output_fmt == "latex":
        return f"\\textbf{{{val}}}"
    return f"*{val}"


Table = list[list[str]]


def table_to_csv_string(table: Table) -> str:
    import csv
    from io import StringIO

    output = StringIO()
    writer = csv.writer(output)
    for row in table:
        writer.writerow(row)
    return output.getvalue().strip()  # Remove trailing newline


def output_table(args, table: Table):
    if args.output_format == "latex":
        args.output_format = "latex_raw"

    if args.output_format == "csv":
        table_str = table_to_csv_string(table)
    else:
        table_str = tabulate(table, headers="firstrow", tablefmt=args.output_format)

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(table_str + "\n")
    else:
        print(table_str + "\n")
