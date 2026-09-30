# Spam Mail Prediction using Machine Learning

## Project Overview
A binary text-classification system that predicts whether an email is **spam** or **ham**. The supplied messages are converted into TF-IDF features and classified with Logistic Regression.

## Dataset
- File: `mail_data.csv`
- Rows: 5572
- Columns: 2 (`Category`, `Message`)
- Class distribution: 4,825 ham and 747 spam

## Methodology
1. Load the supplied mail dataset.
2. Replace missing values with empty strings.
3. Encode `spam` as 0 and `ham` as 1.
4. Split into training and test sets using an 80/20 split with `random_state=3` and stratification.
5. Convert message text into TF-IDF vectors using English stop-word removal and lowercase normalization.
6. Train Logistic Regression.
7. Evaluate using accuracy, precision, recall, F1-score and a confusion matrix.
8. Run a predictive example on a new message.

## Test Results
- Accuracy: 0.9713
- Precision (ham=1): 0.9689
- Recall (ham=1): 0.9990
- F1-score (ham=1): 0.9837

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook Spam_Mail_Prediction.ipynb
```
Or:
```bash
python spam_classifier.py
```

## Files
- `Spam_Mail_Prediction.ipynb` — completed executable notebook
- `spam_classifier.py` — reusable classifier implementation
- `mail_data.csv` — supplied dataset
- `metrics.txt` — measured test metrics
- `README.md` — documentation
- `requirements.txt` — dependencies
