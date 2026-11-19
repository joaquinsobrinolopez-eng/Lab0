# Lab0 - Fundamentals of Continuous Integration

This project is the first lab of the MLOps course. Its goal is to practice the fundamentals of
Continuous Integration (CI): a clean project structure, an isolated virtual environment managed
with `uv`, code linting with `pylint`, automatic code formatting with `black`, and testing with
`pytest` (unit tests and integration tests, with coverage measured by `pytest-cov`).

The project implements a small set of data preprocessing techniques (cleaning, numeric
transformations, text processing and structure utilities) and exposes them through a command
line interface (CLI) built with the `click` package.

## Project structure

```
Lab0/
├── src/
│   ├── __init__.py        # Package marker
│   ├── preprocessing.py   # Logic of the preprocessing functions
│   └── cli.py             # Command line interface (click)
├── tests/
│   ├── test_preprocessing.py  # Unit tests of the logic
│   └── test_cli.py            # Integration tests between the logic and the CLI
├── .gitignore
├── README.md
└── pyproject.toml         # Project configuration and dependencies
```

## Dependencies

| Package      | Purpose                                             |
| ------------ | --------------------------------------------------- |
| Click        | Command line interface                              |
| Black        | Automatic code formatting                           |
| Pylint       | Static analysis and linting                         |
| Pytest       | Testing framework (unit and integration tests)      |
| Pytest-cov   | Coverage measurement of the testing process         |

## Installation

The project uses [uv](https://docs.astral.sh/uv/) to manage the virtual environment and the
dependencies. To replicate the environment, run:

```bash
uv init
uv sync
uv add click black pylint pytest pytest-cov
```

## Usage

Every functionality is available as a command of the `lab0` CLI, organized in four groups:
`clean`, `numeric`, `text` and `struct`.

```bash
# Clean group
uv run lab0 clean remove-missing 1 "" 2 None 3
uv run lab0 clean fill-missing 1 "" 2 --fill-value 9

# Numeric group
uv run lab0 numeric normalize 2 4 6 8 --min 0 --max 1
uv run lab0 numeric standardize 2 4 6 8
uv run lab0 numeric clip 1 2 3 4 --min 2 --max 3
uv run lab0 numeric to-int 1 abc 3.5
uv run lab0 numeric log-transform 1 2.718281828459045 7.38905609893065

# Text group
uv run lab0 text tokenize "Hello, World! This is Lab0."
uv run lab0 text remove-punctuation "Hello, World! 123"
uv run lab0 text remove-stop-words "The cat is here" --stop-words the --stop-words is

# Struct group
uv run lab0 struct shuffle 1 2 3 4 5 --seed 42
uv run lab0 struct flatten "1,2,3" "4,5"
uv run lab0 struct unique 1 1 2 2 3
```

To see the description and an example of any command, use the `--help` flag:

```bash
uv run lab0 --help
uv run lab0 clean --help
uv run lab0 numeric normalize --help
```

## Quality checks

```bash
# Linting with pylint
uv run python -m pylint src/*.py

# Automatic formatting with black
uv run black src/*.py

# Testing with pytest
uv run python -m pytest -v

# Testing with coverage
uv run python -m pytest -v --cov=src
```
