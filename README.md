# AI-Based Loan Approval System

A machine-learning classification project that predicts loan approval outcomes from applicant and loan-application information. The project uses the Kaggle **Loan Prediction Problem** dataset and compares Logistic Regression with a Random Forest classifier.

> **Disclaimer:** This is an educational decision-support demonstration, not a lending or credit-scoring product. Its predictions must not be used to approve or reject real loan applications. The notebook itself highlights the need for further validation, fairness assessment, and documented data provenance.

## Project Objective

- Explore and understand the loan application dataset.
- Clean the target labels and prepare applicant features.
- Train and compare two baseline classification models.
- Evaluate model performance on a held-out test set.
- Inspect Random Forest feature importances.
- Generate a prediction for an illustrative applicant.
- Provide a reusable Python function for predicting loan status.

## Dataset

The notebook downloads the dataset at runtime using `kagglehub`:

- **Dataset:** [Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- **Kaggle dataset identifier:** `altruistdelhite04/loan-prediction-problem-dataset`
- **Target column:** `Loan_Status`

The notebook searches the downloaded CSV files for one containing the `Loan_Status` column and uses that file as the labelled training dataset.

Common dataset fields include applicant demographics, education, employment, income, loan amount, loan term, credit history, and property area. The exact fields are determined by the downloaded dataset.

## Workflow

1. **Download and load data** – Download the Kaggle dataset and locate the labelled CSV.
2. **Exploratory data analysis** – Inspect columns, data types, missing values, duplicate rows, descriptive statistics, and target distribution.
3. **Prepare features and target** – Remove rows with missing target labels, map the binary target to numeric classes, and exclude `Loan_ID` (when present) from the feature matrix.
4. **Split the data** – Use an 80/20 train-test split with stratification and a fixed random seed (`42`).
5. **Preprocess features** – Impute missing values with the most frequent value and one-hot encode categorical values. The preprocessing is fitted within each model pipeline.
6. **Train models** – Fit Logistic Regression and Random Forest classifiers.
7. **Evaluate models** – Compare accuracy, approval-class precision, recall, F1-score, and ROC-AUC. Select the baseline with the highest approval-class F1-score.
8. **Inspect results** – Display a classification report, confusion matrix, and the top 15 Random Forest feature importances.
9. **Make predictions** – Predict an illustrative applicant's loan status and expose a reusable prediction function.

## Models

| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=2000`, `random_state=42` |
| Random Forest | `n_estimators=300`, `min_samples_leaf=2`, `random_state=42`, `n_jobs=-1` |

Both models use scikit-learn pipelines so that preprocessing is applied consistently during training and prediction.

## Evaluation Metrics

The notebook reports:

- **Accuracy** – Overall proportion of correct predictions.
- **Precision (approved)** – Proportion of predicted approvals that are actually approved.
- **Recall (approved)** – Proportion of actual approvals correctly identified.
- **F1 (approved)** – Harmonic mean of approval-class precision and recall; used to select the baseline model.
- **ROC-AUC** – Ranking performance based on predicted class-1 probabilities.

The notebook also displays a classification report and confusion matrix for the selected model. Actual metric values depend on the downloaded data and the train-test split; run the notebook to obtain them.

## Requirements

- Python 3.9 or later (a compatible Python version for the installed library releases)
- Jupyter Notebook or JupyterLab
- Internet access for downloading the dataset through KaggleHub

### Python libraries

- `kagglehub`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `jupyter`

## Setup and Usage

### 1. Clone or download this repository

```bash
git clone https://github.com/rudy0senpai/LoanApproval_model.git
cd LoanApproval_model
```

If you downloaded the project as a ZIP, extract it and open a terminal in the project directory.

### 2. Create and activate a virtual environment (recommended)

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install kagglehub numpy pandas scikit-learn matplotlib seaborn jupyter
```

### 4. Launch Jupyter

```bash
jupyter notebook
```

Open `LoanApproval_model.ipynb` and run the notebook cells from top to bottom.

The first dataset-download run requires internet access. KaggleHub may also require Kaggle access or authentication depending on the dataset's current access settings.

## Reusable Prediction Function

After running the notebook, the function `predict_loan_status()` can accept a dictionary of applicant features using the original dataset column names.

```python
result = predict_loan_status({
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 1500,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1.0,
    "Property_Area": "Urban"
})

print(result)
```

The function returns a dictionary containing:

- `predicted_class` – Numeric predicted class (`0` or `1`).
- `predicted_label` – Original target label according to the mapping created from the dataset.
- `probability_class_1` – Estimated probability of class `1`.

The example above uses common columns from the Kaggle dataset. The input fields should match the columns available in the downloaded training data. Missing columns are reindexed by the notebook's prediction function; for meaningful predictions, provide valid values for the relevant features.

## Repository Structure

```text
.
├── LoanApproval_model.ipynb   # End-to-end analysis, training, evaluation, and prediction
└── README.md                  # Project documentation
```

## Limitations and Responsible Use

- The results are specific to this dataset and the notebook's random train-test split.
- A held-out test score does not establish that a model generalizes to current applicants or real lending conditions.
- The notebook does not establish causal relationships between applicant attributes and loan outcomes.
- Feature importance indicates model reliance, not causation or fairness.
- The illustrative prediction is not a real applicant assessment.
- Before any real-world use, the system would require appropriate data governance, external validation, calibration, threshold selection based on documented costs, subgroup performance and fairness analysis, privacy and legal review, and human oversight.

## Potential Improvements

- Add cross-validation and a separate validation set for model selection.
- Tune the classification threshold using documented business costs.
- Evaluate probability calibration and false-positive / false-negative rates.
- Assess performance across relevant applicant subgroups and investigate fairness risks.
- Add input validation and a user interface only after appropriate model assessment.
- Document dataset provenance and licensing before redistribution.

## References

- [Kaggle – Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- [pandas documentation](https://pandas.pydata.org/docs/)
- [NumPy documentation](https://numpy.org/doc/)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [KaggleHub documentation](https://github.com/Kaggle/kagglehub)
