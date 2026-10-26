# Data note

The dataset is the public Hugging Face dataset [`Krooz/Campus_Recruitment_CSV`](https://huggingface.co/datasets/Krooz/Campus_Recruitment_CSV) (licence recorded in the project as CC0-1.0; origin of the data is not documented).

This repository deliberately **does not include** the held-out test split or any file that contains test rows:

- `data/raw/test.csv`
- `data/processed/test.csv`
- `data/processed/modeling_dataset.csv` and `data/processed/campus_recruitment_combined_flagged.csv` (they contain all rows, including test rows)
- `results/metrics/modeling/08_test_predictions.csv` (per-row test predictions)

Everything else is included: raw train and validation files, processed train and validation splits, all saved models, metrics, figures, notebooks, the Streamlit app and the reports. The test-set results are available as aggregate numbers in `results/metrics/modeling/08_*.csv` and in `report/final_report.md`.

To rebuild the missing files, download the three raw files from Hugging Face into `data/raw/` and re-run `notebooks/02_data_quality_and_split.ipynb` (stratified 70/15/15 split, `random_state=42`). Notebook 08 (the one-time test evaluation) is blocked from re-running by design (`results/metrics/modeling/08_test_access_log.json`). The app (`app/`) never reads the test split.

Study guide: `report/Student_Placement_Predictor_Complete_Project_Guide.pdf`.
