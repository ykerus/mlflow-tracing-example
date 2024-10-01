
# MLflow tracing

## Setting up and running the code

To setup your environment so you can run the code on your local machine, follow the steps below.

1. Make sure you have `uv` installed (see [docs](https://github.com/astral-sh/uv?tab=readme-ov-file#installation))
2. Run `uv sync`
3. Activate the virtual environment: `source .venv/bin/activate` (optional)
4. Run the code: `python src/mlflow_tracing/main.py` (or run `uv run python ...` instead)

## Run MLflow

```bash
mlflow server
```
