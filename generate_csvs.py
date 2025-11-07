import mlflow
import pandas as pd
from pathlib import Path

table_dir = Path("../output/tables")
table_dir.mkdir(parents=True, exist_ok=True)

for dataset in ["devign", "bifi", "reveal"]:
    experiment_name = f"{dataset}_codebert"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment {experiment_name} not found in MLflow!")
        continue

    runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    runs_df.to_csv(table_dir / f"mlflow_{dataset}.csv", index=False)
    print(f"Saved {dataset} → {table_dir / f'mlflow_{dataset}.csv'}")
