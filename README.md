# RR Group Project

## Credit Risk Analysis: Reproducing an R Project in Python

This repository was created for the **Reproducible Research** course final project.  
The project focuses on reproducing an existing R-based data analysis project in Python and making the full workflow transparent, documented, and reproducible.

The main topic of the project is **Credit Risk Analysis** using a loan approval classification dataset.

---

## Group Members

- Ahmet Hasanov
- Kuo Zhang
- Nijat Abiyev
- Yuceltan Ebiri

---

## Course

**Reproducible Research**

---

## Project Objective

The goal of this project is not only to build a machine learning model, but mainly to demonstrate the process of reproduction from R to Python.

The project focuses on:

1. Translating an existing R analysis into Python.
2. Identifying which parts of the original project were easy or difficult to reproduce.
3. Documenting missing information, assumptions, and technical challenges.
4. Creating a clean and user-friendly Python project structure.
5. Making the project fully reproducible through GitHuband `requirements.txt`.

---

## Original Project and Reproduction

The original project was written in R.  
Our task was to reproduce its workflow in Python while keeping the analysis logic as close as possible to the original source.

The Python reproduction includes:

- Data loading
- Exploratory data analysis
- Summary statistics
- Missing value checking
- Data cleaning and outlier handling
- Data visualization
- Categorical variable encoding
- Train-test splitting
- Decision Tree classification
- Random Forest classification
- Model evaluation and comparison

---

## Dataset

The project uses the **Loan Approval Classification Data** dataset from Kaggle.

Dataset link:  
https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data

The dataset is stored in the project under:

```text
data/loan_data.csv
```

---

## Project Structure

```text
RR_groupproject/
│
├── README.md
├── requirements.txt
├── main.py
├── loan.py
├── loan_pro.ipynb
│
├── data/
│   └── loan_data.csv
│
├── outputs/
│   └── figures/
│
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── explorer.py
    ├── cleaner.py
    ├── visualizer.py
    ├── preprocessor.py
    └── models.py
```

---

## File Descriptions

| File / Folder | Description |
|---|---|
| `loan.py` | Initial direct Python translation of the R project |
| `loan_pro.ipynb` | Notebook version used for development and checking results |
| `main.py` | Main script that runs the complete modular Python workflow |
| `src/data_loader.py` | Loads the loan dataset |
| `src/explorer.py` | Provides basic data inspection, summaries, and missing value checks |
| `src/cleaner.py` | Handles data cleaning and outlier treatment |
| `src/visualizer.py` | Creates visualizations and saves figures |
| `src/preprocessor.py` | Encodes categorical variables and prepares train-test splits |
| `src/models.py` | Trains and evaluates the Decision Tree and Random Forest models |
| `data/` | Contains the dataset |
| `outputs/figures/` | Stores generated plots |
| `requirements.txt` | Lists the Python packages needed to run the project |

---

## How to Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/Kuo629/RR_groupproject.git
cd RR_groupproject
```

### 2. Create and activate a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install required packages

```bash
pip install -r requirements.txt
```

### 4. Run the project

```bash
python main.py
```

---


## Required Software

The project was developed using:

- Python 3.11
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Exact package versions can be found in `requirements.txt`.

---

## Reproduction Process

The original R project was reviewed and then translated step by step into Python.

The reproduction followed these stages:

1. Understanding the original R workflow.
2. Recreating the data loading process in Python.
3. Translating summary statistics and exploratory data analysis.
4. Recreating the original visualizations using Python libraries.
5. Applying similar cleaning rules for outliers and missing values.
6. Encoding categorical variables for machine learning models.
7. Rebuilding the Decision Tree model in Python.
8. Rebuilding the Random Forest model in Python.
9. Comparing the model outputs and final results.
10. Refactoring the code into a modular and reusable structure.

---

## Challenges During Reproduction

During the reproduction process, we faced several challenges:

- Some R visualization functions did not have exact one-to-one equivalents in Python.
- Plot formatting had to be adjusted manually using `matplotlib` and `seaborn`.
- Categorical variables required explicit encoding before model training.
- Missing values had to be handled carefully, especially before training the Random Forest model.
- The original workflow was mostly script-based, so we reorganized it into separate Python modules.
- Plots were saved as files so the results could be reviewed after running the script.

---

## What Was Missing in the Original Source

The reproduction process showed that the original project could have been easier to reproduce if it had included:

- A clearer folder structure
- A dependency file listing software and package versions
- More detailed comments explaining each analytical step
- A single command for running the full analysis
- Clearer separation between data cleaning, visualization, and modeling
- More detailed documentation about the dataset and assumptions

---

## Models Used

Two classification models were implemented:

### Decision Tree Classifier

The Decision Tree model was used as a simple and interpretable classification model.

### Random Forest Classifier

The Random Forest model was used as an ensemble method to improve predictive performance and compare results against the Decision Tree model.

The final comparison is based on classification metrics such as:

- Confusion matrix
- Accuracy
- Classification report
- Feature importance

---

## Outputs

The project generates:

- Printed data summaries
- Missing value reports
- Exploratory plots
- Model evaluation results
- Confusion matrices
- Feature importance plots

Generated figures are saved under:

```text
outputs/figures/
```

---

## Team Contributions

| Team Member | Contribution |
|---|---|
| Ahmet Hasanov | Documentation support, review of reproducibility process, presentation preparation |
| Kuo Zhang | Repository setup, GitHub management, project coordination |
| Nijat Abiyev | Dataset review, testing, and result checking |
| Yuceltan Ebiri | Python translation, modular code structure, machine learning implementation, environment setup |

---

## Version Control

Git and GitHub were used throughout the project to track changes and collaboration.

The repository history includes updates related to:

- Initial project setup
- Python translation
- Modular code organization
- Dataset integration
- Documentation improvements
- Final reproducibility preparation

---

## Project Status

The project is prepared as a reproducible Python version of the original R analysis.  
The current version includes code, documentation, dataset handling, model implementation, and local environment setup.

---

## License

This project was prepared for educational purposes as part of the Reproducible Research course.
