"""Integration tests between the logic and the command line interface.

The tests invoke the commands of the ``lab0`` CLI with the ``CliRunner``
provided by click and check that the printed result matches the output of
the preprocessing logic used by each command.
"""

import pytest
from click.testing import CliRunner

from src import cli as cli_module


@pytest.fixture
def runner():
    """Return a CliRunner instance shared by all the CLI tests."""
    return CliRunner()


def test_clean_remove_missing(runner):
    """It removes the missing values given as string arguments."""
    result = runner.invoke(
        cli_module.cli, ["clean", "remove-missing", "1", "", "2", "None", "3"]
    )
    assert result.exit_code == 0
    assert "Result: ['1', '2', '3']" in result.output


def test_clean_fill_missing(runner):
    """It fills the missing values with the value of the option."""
    result = runner.invoke(
        cli_module.cli, ["clean", "fill-missing", "1", "", "2", "--fill-value", "9"]
    )
    assert result.exit_code == 0
    assert "Result: ['1', '9', '2']" in result.output


def test_numeric_normalize(runner):
    """It scales the numbers to the default range from 0 to 1."""
    result = runner.invoke(cli_module.cli, ["numeric", "normalize", "2", "4", "6", "8"])
    assert result.exit_code == 0
    assert "Result: [0.0, 0.3333333333333333, 0.6666666666666666, 1.0]" in result.output


def test_numeric_normalize_new_range(runner):
    """It scales the numbers to the range given by the options."""
    result = runner.invoke(
        cli_module.cli, ["numeric", "normalize", "2", "4", "6", "8", "--min", "0", "--max", "10"]
    )
    assert result.exit_code == 0
    assert "Result: [0.0, 3.333333333333333, 6.666666666666666, 10.0]" in result.output


def test_numeric_standardize(runner):
    """It standardizes the numbers with the z-score method."""
    result = runner.invoke(cli_module.cli, ["numeric", "standardize", "2", "4", "6", "8"])
    assert result.exit_code == 0
    assert "-1.3416" in result.output
    assert "0.4472" in result.output


def test_numeric_clip(runner):
    """It clips the numbers to the range given by the options."""
    result = runner.invoke(
        cli_module.cli, ["numeric", "clip", "1", "2", "3", "4", "--min", "2", "--max", "3"]
    )
    assert result.exit_code == 0
    assert "Result: [2.0, 2.0, 3.0, 3.0]" in result.output


def test_numeric_convert_to_int(runner):
    """It converts the numerical values to integers and excludes the rest."""
    result = runner.invoke(cli_module.cli, ["numeric", "to-int", "1", "abc", "3.5"])
    assert result.exit_code == 0
    assert "Result: [1, 3]" in result.output


def test_numeric_log_transform(runner):
    """It applies the natural log to the positive values."""
    result = runner.invoke(
        cli_module.cli, ["numeric", "log-transform", "1", "2.718281828459045", "7.38905609893065"]
    )
    assert result.exit_code == 0
    assert "Result: [0.0, 1.0, 2.0]" in result.output


def test_text_tokenize(runner):
    """It tokenizes the text into lower-cased alphanumeric words."""
    result = runner.invoke(cli_module.cli, ["text", "tokenize", "Hello, World! This is Lab0."])
    assert result.exit_code == 0
    assert "Result: hello world this is lab0" in result.output


def test_text_remove_punctuation(runner):
    """It keeps only alphanumerical characters and spaces."""
    result = runner.invoke(cli_module.cli, ["text", "remove-punctuation", "Hello, World! 123"])
    assert result.exit_code == 0
    assert "Result: Hello World 123" in result.output


def test_text_remove_stop_words(runner):
    """It removes the stop words given through the option."""
    result = runner.invoke(
        cli_module.cli,
        [
            "text",
            "remove-stop-words",
            "The cat is on the mat",
            "--stop-words",
            "the",
            "--stop-words",
            "is",
            "--stop-words",
            "on",
        ],
    )
    assert result.exit_code == 0
    assert "Result: cat mat" in result.output


def test_struct_shuffle(runner):
    """It shuffles the values in a reproducible way with the seed."""
    result = runner.invoke(
        cli_module.cli, ["struct", "shuffle", "a", "b", "c", "d", "e", "--seed", "42"]
    )
    assert result.exit_code == 0
    assert "Result: ['d', 'b', 'c', 'e', 'a']" in result.output


def test_struct_shuffle_without_seed(runner):
    """It shuffles the values keeping the same elements."""
    result = runner.invoke(cli_module.cli, ["struct", "shuffle", "1", "2", "3", "4", "5"])
    assert result.exit_code == 0
    assert "1" in result.output and "5" in result.output


def test_struct_flatten(runner):
    """It flattens the comma separated sublists into one list."""
    result = runner.invoke(cli_module.cli, ["struct", "flatten", "1.5,2", "x,y"])
    assert result.exit_code == 0
    assert "Result: [1.5, 2, 'x', 'y']" in result.output


def test_struct_unique(runner):
    """It removes the duplicated values keeping the original order."""
    result = runner.invoke(cli_module.cli, ["struct", "unique", "1", "1", "2", "2", "3"])
    assert result.exit_code == 0
    assert "Result: [1, 2, 3]" in result.output


def test_numeric_normalize_without_values(runner):
    """It fails with a usage error when no values are provided."""
    result = runner.invoke(cli_module.cli, ["numeric", "normalize"])
    assert result.exit_code != 0
