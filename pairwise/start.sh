#!/usr/bin/env bash
set -e
uvicorn app.api:app --host 0.0.0.0 --port "${API_PORT:-8000}" &
exec streamlit run app/ui.py --server.address 0.0.0.0 --server.port "${PORT:-8501}"
