
# BioDataLab

**An open-source toolkit for biological, agricultural and experimental data analysis.**

BioDataLab is a Python-based platform designed to simplify the analysis, validation, cleaning and visualization of scientific datasets, particularly in biological and agricultural research.

## Features

- **Data Cleaning** — Handle missing values, remove empty columns and prepare datasets for analysis.
- **Data Validation** — Validate datasets against configurable rules, data types, ranges and uniqueness constraints.
- **Data Quality Assessment** — Evaluate dataset quality through structured indicators and reports.
- **Statistical Analysis** — Perform statistical analyses and explore experimental data.
- **Visualization** — Generate informative graphical representations of datasets.
- **Regression Analysis** — Explore relationships between variables using regression methods.
- **Group Comparisons** — Compare experimental groups using statistical methods.
- **Data Export** — Export processed data and analytical results.
- **Automated Testing** — Maintain reliability through automated tests and continuous integration.

## Installation

Clone the repository:

```bash
git clone https://github.com/antoinejaquot-max/biodatalab.git
cd biodatalab
```

Install the package:

```bash
pip install .
```

## Usage

BioDataLab provides modular tools for processing and analyzing scientific datasets using Python and pandas.

```python
import pandas as pd
from biodatalab import validate_dataframe

data = pd.DataFrame({
    "temperature": [20.1, 21.3, 19.8],
    "growth": [1.2, 1.5, 1.1],
})

result = validate_dataframe(data)

print(result)
```

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

## Project Goals

BioDataLab aims to provide accessible, reproducible and modular analytical tools for biological and agricultural research, with an emphasis on data reliability, scientific usability and transparent workflows.

## License

This project is distributed under the license specified in the `LICENSE` file.
