# Modular fraud detection pipeline

Copy the `fraud_pipeline` folder to the root of your existing project, next to `configs/`. Keep your current `configs/training_config.json` and virtual environment.

Run from `E:\\fraud detection`:

```bash
venv\\Scripts\\activate
python -m fraud_pipeline.train
```

## Modules
- `config.py`: central configuration.
- `data.py`: ingestion, column normalization, validation and stratified splitting.
- `preprocessing.py`: imputation, scaling, one-hot encoding and preprocessed CSV export.
- `models.py`: model search spaces and estimator reconstruction.
- `threshold.py`: positive-class scores and validation threshold selection.
- `training.py`: Optuna tuning and validation-based model selection.
- `evaluation.py`: test evaluation, Joblib export and MLflow registration.
- `train.py`: orchestration entry point.

Run from the project root so `configs/training_config.json` resolves correctly. Keep the monolithic script as a backup until the modular pipeline completes an end-to-end test.
