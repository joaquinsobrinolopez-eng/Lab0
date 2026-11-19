"""Data preprocessing techniques for different types of data.

This module contains the logic of the project: a set of functions to clean,
transform and reorganize lists of values and text. The command line interface
defined in ``cli.py`` uses these functions.
"""

import math
import random
import re
import statistics

MISSING_STRINGS = ("none", "nan")


def _is_missing(value):
    """Check whether a value is considered as missing.

    A value is missing when it is ``None``, an empty string, a float ``nan``,
    or one of the common string representations of missing values (``"None"``
    and ``"nan"``). This makes the function useful both for Python objects and
    for the string values received from the command line.

    Args:
        value: The value to check.

    Returns:
        True if the value is missing, False otherwise.
    """
    if value is None or value == "":
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip().lower() in MISSING_STRINGS


def remove_missing_values(values):
    """Remove the missing values of a list of values.

    Args:
        values: List of values, including missing values like None, '' and nan.

    Returns:
        List of values without the missing ones.
    """
    return [value for value in values if not _is_missing(value)]


def fill_missing_values(values, fill_value=0):
    """Replace the missing values of a list with the given filling value.

    Args:
        values: List of values, including missing values.
        fill_value: Value used to replace the missing values. Defaults to 0.

    Returns:
        List of values where the missing values are replaced with fill_value.
    """
    return [fill_value if _is_missing(value) else value for value in values]


def remove_duplicates(values):
    """Remove the duplicated values of a list, keeping the original order.

    Args:
        values: List of values, possibly with duplicates.

    Returns:
        List of unique values in the same order as they first appear.
    """
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def min_max_normalize(values, new_min=0.0, new_max=1.0):
    """Normalize a list of numerical values using the min-max method.

    Args:
        values: List of numerical values.
        new_min: Minimum of the new range. Defaults to 0.0.
        new_max: Maximum of the new range. Defaults to 1.0.

    Returns:
        List of values scaled to the new range. When all the original values
        are equal, every value is mapped to the new minimum.
    """
    old_min = min(values)
    old_max = max(values)
    if old_max == old_min:
        return [new_min for _ in values]
    old_range = old_max - old_min
    new_range = new_max - new_min
    return [(value - old_min) / old_range * new_range + new_min for value in values]


def z_score_standardize(values):
    """Standardize a list of numerical values using the z-score method.

    Args:
        values: List of numerical values.

    Returns:
        List of values centered on their mean and scaled by their standard
        deviation. When all the values are equal, every value becomes 0.0.
    """
    mean = statistics.mean(values)
    std = statistics.pstdev(values)
    if std == 0:
        return [0.0 for _ in values]
    return [(value - mean) / std for value in values]


def clip_values(values, min_value, max_value):
    """Clip a list of numerical values to the given range.

    Args:
        values: List of numerical values.
        min_value: Minimum value of the range.
        max_value: Maximum value of the range.

    Returns:
        List of values where every value smaller than min_value becomes
        min_value and every value greater than max_value becomes max_value.
    """
    return [max(min(value, max_value), min_value) for value in values]


def convert_to_int(values):
    """Convert the values of a list of strings to integers.

    Non-numerical values are excluded from the output. Values with a decimal
    part are truncated, so ``"3.5"`` becomes ``3``.

    Args:
        values: List of strings, both numerical and non-numerical.

    Returns:
        List with only the values that could be converted to integers.
    """
    result = []
    for value in values:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            try:
                result.append(int(float(value)))
            except (TypeError, ValueError):
                continue
    return result


def log_transform(values):
    """Transform the positive values of a list to logarithmic scale.

    Args:
        values: List of numerical values.

    Returns:
        List with the natural logarithm of the original positive values.
    """
    return [math.log(value) for value in values if value > 0]


def tokenize_text(text):
    """Tokenize a text into words of alphanumeric characters, lower-cased.

    Args:
        text: Text to be processed.

    Returns:
        String with the tokens of the text separated by single spaces. Only
        alphanumeric characters are kept and all the words are lower-cased.
    """
    words = text.lower().split()
    tokens = [re.sub(r"[^a-z0-9]", "", word) for word in words]
    return " ".join(token for token in tokens if token)


def remove_punctuation(text):
    """Keep only the alphanumerical characters and the spaces of a text.

    Args:
        text: Text to be processed.

    Returns:
        Text without any character that is not alphanumeric or a space.
    """
    return re.sub(r"[^a-zA-Z0-9 ]", "", text)


def remove_stop_words(text, stop_words):
    """Remove the stop words of a text, which is first lower-cased.

    Args:
        text: Text to be processed.
        stop_words: List of stop words to be removed.

    Returns:
        Lower-cased text without the words contained in stop_words.
    """
    stop_words_lower = {word.lower() for word in stop_words}
    words = text.lower().split()
    return " ".join(word for word in words if word not in stop_words_lower)


def flatten_list(values):
    """Flatten a list of lists into a single list.

    Args:
        values: A list of lists.

    Returns:
        List with all the elements of the sublists in their original order.
    """
    result = []
    for sublist in values:
        result.extend(sublist)
    return result


def shuffle_list(values, seed=None):
    """Randomly shuffle a list of values.

    Args:
        values: List of values.
        seed: Seed of the random generator, to ensure reproducibility.
            Defaults to None, which produces a different order every run.

    Returns:
        New list with the same values in random order.
    """
    shuffled = list(values)
    random.Random(seed).shuffle(shuffled)
    return shuffled
