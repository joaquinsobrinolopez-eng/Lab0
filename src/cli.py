"""Command line interface of the preprocessing functionalities.

The interface is built with the ``click`` package. A general group named
``cli`` contains four groups: ``clean``, ``numeric``, ``text`` and ``struct``.
Each command wraps one function of the ``preprocessing`` module and prints
the result of applying it to the values received from the command line.
"""

import click

from src import preprocessing


def _parse_value(value):
    """Convert a string value to int or float when possible.

    Args:
        value: String value received from the command line.

    Returns:
        The value converted to int or float when it represents a number, or
        the original string otherwise.
    """
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _parse_list(values):
    """Convert a tuple of string values to a list of numbers or strings.

    Args:
        values: Tuple of string values received from the command line.

    Returns:
        List with each value converted to int or float when possible.
    """
    return [_parse_value(value) for value in values]


@click.group()
def cli():
    """Command line interface for the data preprocessing functionalities."""


@cli.group()
def clean():
    """Commands to clean data, like missing values."""


@clean.command("remove-missing")
@click.argument("values", nargs=-1, required=True, type=str)
def remove_missing(values):
    """Remove the missing values (None, '' or nan) of a list of values.

    Example:
        uv run lab0 clean remove-missing 1 "" 2 None 3
    """
    result = preprocessing.remove_missing_values(list(values))
    click.echo(f"Result: {result}")


@clean.command("fill-missing")
@click.argument("values", nargs=-1, required=True, type=str)
@click.option(
    "--fill-value",
    default="0",
    type=str,
    help="Value used to replace the missing values (defaults to 0).",
)
def fill_missing(values, fill_value):
    """Replace the missing values of a list with the given filling value.

    Example:
        uv run lab0 clean fill-missing 1 "" 2 --fill-value 9
    """
    result = preprocessing.fill_missing_values(list(values), fill_value)
    click.echo(f"Result: {result}")


@cli.group()
def numeric():
    """Commands to deal with numerical attributes."""


@numeric.command()
@click.argument("values", nargs=-1, required=True, type=float)
@click.option("--min", "new_min", type=float, default=0.0, help="New minimum value.")
@click.option("--max", "new_max", type=float, default=1.0, help="New maximum value.")
def normalize(values, new_min, new_max):
    """Normalize a list of numbers to a new range using min-max.

    Example:
        uv run lab0 numeric normalize 2 4 6 8 --min 0 --max 1
    """
    result = preprocessing.min_max_normalize(list(values), new_min, new_max)
    click.echo(f"Result: {result}")


@numeric.command()
@click.argument("values", nargs=-1, required=True, type=float)
def standardize(values):
    """Standardize a list of numbers using the z-score method.

    Example:
        uv run lab0 numeric standardize 2 4 6 8
    """
    result = preprocessing.z_score_standardize(list(values))
    click.echo(f"Result: {result}")


@numeric.command()
@click.argument("values", nargs=-1, required=True, type=float)
@click.option(
    "--min", "min_value", type=float, default=0.0, help="Minimum value to clip."
)
@click.option(
    "--max", "max_value", type=float, default=1.0, help="Maximum value to clip."
)
def clip(values, min_value, max_value):
    """Clip a list of numbers to the given range.

    Example:
        uv run lab0 numeric clip 1 2 3 4 --min 2 --max 3
    """
    result = preprocessing.clip_values(list(values), min_value, max_value)
    click.echo(f"Result: {result}")


@numeric.command("to-int")
@click.argument("values", nargs=-1, required=True, type=str)
def convert_to_int(values):
    """Convert the numerical values of a list to integers.

    Non-numerical values are excluded from the output.

    Example:
        uv run lab0 numeric to-int 1 abc 3.5
    """
    result = preprocessing.convert_to_int(list(values))
    click.echo(f"Result: {result}")


@numeric.command("log-transform")
@click.argument("values", nargs=-1, required=True, type=float)
def log_transform(values):
    """Transform the positive values of a list to logarithmic scale.

    Example:
        uv run lab0 numeric log-transform 1 2.718281828459045 7.38905609893065
    """
    result = preprocessing.log_transform(list(values))
    click.echo(f"Result: {result}")


@cli.group()
def text():
    """Commands to deal with textual information."""


@text.command()
@click.argument("input_text", type=str)
def tokenize(input_text):
    """Tokenize a text into alphanumeric words, lower-cased.

    Example:
        uv run lab0 text tokenize "Hello, World! This is Lab0."
    """
    result = preprocessing.tokenize_text(input_text)
    click.echo(f"Result: {result}")


@text.command("remove-punctuation")
@click.argument("input_text", type=str)
def remove_punctuation(input_text):
    """Keep only the alphanumerical characters and spaces of a text.

    Example:
        uv run lab0 text remove-punctuation "Hello, World! 123"
    """
    result = preprocessing.remove_punctuation(input_text)
    click.echo(f"Result: {result}")


@text.command("remove-stop-words")
@click.argument("input_text", type=str)
@click.option(
    "--stop-words",
    type=str,
    multiple=True,
    help="Stop words to be removed from the text.",
)
def remove_stop_words(input_text, stop_words):
    """Remove the stop words of a text, which is first lower-cased.

    Example:
        uv run lab0 text remove-stop-words "The cat is here" --stop-words the --stop-words is
    """
    result = preprocessing.remove_stop_words(input_text, list(stop_words))
    click.echo(f"Result: {result}")


@cli.group()
def struct():
    """Commands related to the structure of the data."""


@struct.command()
@click.argument("values", nargs=-1, required=True, type=str)
@click.option("--seed", type=int, default=None, help="Seed for reproducibility.")
def shuffle(values, seed):
    """Randomly shuffle a list of values.

    Example:
        uv run lab0 struct shuffle 1 2 3 4 5 --seed 42
    """
    result = preprocessing.shuffle_list(_parse_list(values), seed)
    click.echo(f"Result: {result}")


@struct.command()
@click.argument("values", nargs=-1, required=True, type=str)
def flatten(values):
    """Flatten a list of lists into a single list.

    Each argument is one sublist, and its elements are separated by commas.

    Example:
        uv run lab0 struct flatten "1,2,3" "4,5"
    """
    nested = [_parse_list(value.split(",")) for value in values]
    result = preprocessing.flatten_list(nested)
    click.echo(f"Result: {result}")


@struct.command()
@click.argument("values", nargs=-1, required=True, type=str)
def unique(values):
    """Remove the duplicated values of a list, keeping the original order.

    Example:
        uv run lab0 struct unique 1 1 2 2 3
    """
    result = preprocessing.remove_duplicates(_parse_list(values))
    click.echo(f"Result: {result}")
