"""Unit tests of the data preprocessing logic."""

import math

import pytest

from src import preprocessing


@pytest.fixture
def number_list():
    """Return a list of numbers that can be reused by several tests."""
    return [2.0, 4.0, 6.0, 8.0]


@pytest.mark.parametrize(
    "values, expected",
    [
        ([1, None, 2], [1, 2]),
        (["", "a", ""], ["a"]),
        ([float("nan"), 3.0], [3.0]),
        (["nan", "None", "x"], ["x"]),
    ],
)
def test_remove_missing_values(values, expected):
    """It removes None, empty strings, nan and their string forms."""
    assert preprocessing.remove_missing_values(values) == expected


@pytest.mark.parametrize(
    "values, fill_value, expected",
    [
        ([1, None, 2], 0, [1, 0, 2]),
        ([1, None, 2], 9, [1, 9, 2]),
        (["a", "", "b"], "x", ["a", "x", "b"]),
        (["a", "nan", "b"], "x", ["a", "x", "b"]),
    ],
)
def test_fill_missing_values(values, fill_value, expected):
    """It replaces the missing values with the given filling value."""
    assert preprocessing.fill_missing_values(values, fill_value) == expected


def test_fill_missing_values_default_fill_value():
    """It uses 0 as the default filling value."""
    assert preprocessing.fill_missing_values([1, None, 2]) == [1, 0, 2]


@pytest.mark.parametrize(
    "values, expected",
    [
        ([1, 1, 2, 2, 3], [1, 2, 3]),
        (["a", "b", "a", "c"], ["a", "b", "c"]),
        ([1, 2, 3], [1, 2, 3]),
    ],
)
def test_remove_duplicates(values, expected):
    """It keeps only the first occurrence of every value."""
    assert preprocessing.remove_duplicates(values) == expected


def test_min_max_normalize(number_list):
    """It scales the numbers to the default range from 0.0 to 1.0."""
    result = preprocessing.min_max_normalize(number_list)
    assert result == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])


def test_min_max_normalize_new_range(number_list):
    """It scales the numbers to the new range given by the options."""
    result = preprocessing.min_max_normalize(number_list, new_min=-1.0, new_max=1.0)
    assert result == pytest.approx([-1.0, -1 / 3, 1 / 3, 1.0])


def test_min_max_normalize_constant_values():
    """It maps a constant list to the new minimum."""
    assert preprocessing.min_max_normalize([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]


def test_z_score_standardize(number_list):
    """It centers the numbers on their mean and divides by their std."""
    std = math.sqrt(5)
    result = preprocessing.z_score_standardize(number_list)
    assert result == pytest.approx([-3 / std, -1 / std, 1 / std, 3 / std])


def test_z_score_standardize_constant_values():
    """It returns zeros when all the values are equal."""
    assert preprocessing.z_score_standardize([3.0, 3.0, 3.0]) == [0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "values, min_value, max_value, expected",
    [
        ([1, 2, 3, 4], 2, 3, [2, 2, 3, 3]),
        ([-5, 0, 5], -1, 1, [-1, 0, 1]),
        ([0.5, 1.5], 0, 1, [0.5, 1.0]),
    ],
)
def test_clip_values(values, min_value, max_value, expected):
    """It limits every value to the given minimum and maximum."""
    assert preprocessing.clip_values(values, min_value, max_value) == expected


@pytest.mark.parametrize(
    "values, expected",
    [
        (["1", "2", "3"], [1, 2, 3]),
        (["1", "abc", "3.5"], [1, 3]),
        ([None, "2"], [2]),
        (["x", "y"], []),
    ],
)
def test_convert_to_int(values, expected):
    """It converts the numerical values and excludes the rest."""
    assert preprocessing.convert_to_int(values) == expected


def test_log_transform():
    """It keeps only the positive values and applies the natural log."""
    values = [1, math.e, 7.38905609893065, -2, 0]
    assert preprocessing.log_transform(values) == pytest.approx([0.0, 1.0, 2.0])


def test_tokenize_text():
    """It lower-cases the words and keeps only alphanumeric characters."""
    assert preprocessing.tokenize_text("Hello, World! This is Lab0.") == (
        "hello world this is lab0"
    )
    assert preprocessing.tokenize_text("don't stop") == "dont stop"


def test_remove_punctuation():
    """It keeps only alphanumerical characters and spaces."""
    assert preprocessing.remove_punctuation("Hello, World! 123") == "Hello World 123"
    assert preprocessing.remove_punctuation("a-b_c?!") == "abc"


@pytest.mark.parametrize(
    "text, stop_words, expected",
    [
        ("The cat is on the mat", ["the", "is", "on"], "cat mat"),
        ("Hello World", [], "hello world"),
        ("A B C", ["b"], "a c"),
    ],
)
def test_remove_stop_words(text, stop_words, expected):
    """It lower-cases the text and removes the given stop words."""
    assert preprocessing.remove_stop_words(text, stop_words) == expected


def test_flatten_list():
    """It joins all the sublists into a single flat list."""
    assert preprocessing.flatten_list([[1, 2], [3], [4, 5]]) == [1, 2, 3, 4, 5]


def test_shuffle_list():
    """It returns a new shuffled list and does not modify the original."""
    values = [1, 2, 3, 4, 5]
    result = preprocessing.shuffle_list(values, seed=42)
    assert result == [4, 2, 3, 5, 1]
    assert values == [1, 2, 3, 4, 5]


def test_shuffle_list_reproducible():
    """The same seed always produces the same order."""
    values = ["a", "b", "c", "d", "e", "f"]
    first = preprocessing.shuffle_list(values, seed=7)
    second = preprocessing.shuffle_list(values, seed=7)
    assert first == second
    assert sorted(first) == sorted(values)

