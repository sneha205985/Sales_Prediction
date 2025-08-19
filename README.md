## Sales Prediction with Advertising Data

This project demonstrates an end-to-end machine learning workflow in Python to predict product sales from advertising spend across TV, Radio, and Newspaper channels. It includes data loading, model training with cross-validated model selection, evaluation, and a simple CLI for batch or single prediction.

### Dataset
- Location: `data/advertising.csv`
- Columns: `TV`, `Radio`, `Newspaper`, `Sales`

### Quickstart
1. Create and activate a virtual environment (recommended):
```bash
python3 -m venv .venv && source .venv/bin/activate
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Train a model
Run cross-validated training and save the best model and a report to `models/`:
```bash
python -m sales_prediction.cli.train \
  --csv-path data/advertising.csv \
  --target-column Sales \
  --output-dir models \
  --cv 5 \
  --test-size 0.2 \
  --random-state 42
```

Outputs created in `models/`:
- `best_model.joblib`: the fitted best pipeline
- `model_report.json`: metrics, chosen model, hyperparameters, and feature names

### Make predictions
After training, you can predict from a CSV file containing `TV,Radio,Newspaper` columns:
```bash
python -m sales_prediction.cli.predict \
  --model-path models/best_model.joblib \
  --report-path models/model_report.json \
  --csv-input data/advertising.csv \
  --out predictions.csv
```

Or predict for a single configuration:
```bash
python -m sales_prediction.cli.predict \
  --model-path models/best_model.joblib \
  --report-path models/model_report.json \
  --tv 230.1 --radio 37.8 --newspaper 69.2
```

### Project layout
- `data/advertising.csv`: sample dataset
- `src/sales_prediction/`: reusable library code
  - `data.py`: data loading helpers
  - `modeling.py`: model building, training, evaluation, and artifact saving
  - `cli/train.py`: training command
  - `cli/predict.py`: prediction command

### Notes
- Models evaluated: Linear Regression, Ridge, Lasso, and Random Forest Regressor.
- Selection metric: Root Mean Squared Error (via cross-validation on the training split).
- The CLI stores feature names to ensure consistent prediction ordering.

