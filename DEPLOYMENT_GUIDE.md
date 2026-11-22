# Streamlit Community Cloud Deployment Guide

## Deployment status

The project is prepared for Streamlit Community Cloud. No public URL has been
created or verified because this workspace has no authenticated GitHub or
Streamlit deployment account access.

## Required files

- `app.py`
- `requirements.txt`
- `runtime.txt`
- `.streamlit/config.toml`
- `configs/project.json`
- `src/` and `app/`
- `data/raw/Palo Alto Networks.csv`
- Required `data/processed/*.joblib` and risk CSV artifacts
- Required `models/**/*.joblib` artifacts
- Required `reports/*.csv`, `reports/*.json`, and SHAP/fairness reports

Do not remove the raw CSV or the Phase 5/6 model artifacts. The dashboard loads
historical data and existing trained/calibrated artifacts; it does not retrain.

## Python and dependencies

- Deployment runtime: Python 3.12.10
- Streamlit entry point: `app.py`
- No API key or secret is currently required.
- The pinned dependencies are tested for Python 3.13 locally and suitable for
  deployment on Python 3.12.

## GitHub preparation

1. Initialize Git if needed:

   ```powershell
   git init
   git add .
   git status
   ```

2. Confirm that `.env`, `.venv`, caches, credentials, and private local files
   are not listed.
3. Confirm that `app.py`, the data CSV, required model joblib files, and required
   reports are listed.
4. Commit and push to GitHub.

The repository currently excludes no required model or report files. Do not force
unknown private datasets into Git.

## Streamlit Community Cloud setup

1. Sign in to Streamlit Community Cloud.
2. Create a new app from the GitHub repository.
3. Select the repository and branch.
4. Set main file path to `app.py`.
5. Use the repository `runtime.txt` and `requirements.txt`.
6. Leave the secrets section empty unless an external service is intentionally added.
7. Deploy.

## Secrets

The current application has no secrets, API keys, database credentials, or
external authenticated services. Do not add secrets unless later integration
actually requires them.

## Local smoke test

```powershell
.\\.venv\\Scripts\\python.exe -m streamlit run app.py --server.headless=true --server.address=127.0.0.1 --server.port=8501
curl.exe http://127.0.0.1:8501
```

Stop the local server after verification.

## Artifact verification

```powershell
python scripts/verify_deployment.py
python scripts/verify_artifacts.py
python -m pytest tests/test_phase12_deployment.py -q
```

## Troubleshooting

- **Missing module:** compare Linux package names and rerun `pip install -r requirements.txt`.
- **Missing artifact:** check the GitHub file list and case-sensitive path.
- **Model loading error:** verify the joblib file was committed and is under 25 MB.
- **CSV not found:** confirm `data/raw/Palo Alto Networks.csv` is tracked.
- **Sensitive fields:** Age, Gender, and MaritalStatus must remain audit-only.
- **Streamlit start error:** confirm `app.py` is the main file and Python 3.12 is selected.

## Redeployment

Push source or artifact changes to the connected branch. Streamlit normally
redeploys automatically. Re-run artifact and dashboard smoke checks after every
model or report replacement.
