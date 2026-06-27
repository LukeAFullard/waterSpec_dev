# Chapter 2: Installation and Setup

## 2.1 Requirements & Dependencies

**waterSpec** requires Python 3.8 or higher.

The package relies on a robust ecosystem of core data science dependencies: **numpy**, **scipy**, **pandas**, and **matplotlib**.

Additionally, the advanced analytical capabilities are powered by specialized dependencies:
*   **astropy** (for Lomb-Scargle periodograms)
*   **ruptures** (for changepoint detection)
*   **mannks** and **piecewise-regression** (for robust segmented model fitting)

You do not need to install these manually; they will be installed automatically when you install **waterSpec** using **pip**.

## 2.2 Installation Methods

To install the package directly from GitHub, follow these step-by-step instructions.

1.  Open your terminal.
2.  Clone the repository using `git clone`.
3.  Navigate into the downloaded directory.
4.  Install the package using **pip**.

Run the following commands to perform a standard installation:

```bash
git clone https://github.com/LukeAFullard/waterSpec_dev.git
cd waterSpec_dev
pip install -e .
```

The command `pip install -e .` installs the package in "editable" mode. This is great for researchers who might want to tweak the code, as any modifications you make to the source files will immediately be reflected without needing to reinstall.

If you plan to develop or run tests, you can perform a development/testing installation which includes the **pytest** suite:

```bash
pip install -e '.[test]'
```

## 2.3 Validating the Installation

After installing, it is highly recommended to verify that the package installed correctly.

First, you can test the import with a quick Python snippet. Open a Python shell and run:

```python
import waterSpec
print("waterSpec installed successfully!")
```

Second, if you installed the testing dependencies, you can run the full test suite to ensure all statistical functions are working correctly on your machine. Open your terminal and run:

```bash
pytest tests/
```

Seeing "passed" for the tests confirms the mathematical integrity of the installation.
