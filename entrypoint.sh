#!/bin/sh
set -e
# Fail on any error


if [ "$ENV_TYPE" = "dev" ]; then
    echo "[ENV: DEV] Starting FastAPI + Streamlit..."

    # Start FastAPI server:
    cd /fastAPI
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload &

    # Start Streamlit app: 
    cd /streamlit
    streamlit run app.py --server.port 8501
else
    echo "[ENV: PROD] Starting FastAPI only..."
    cd /fastAPI
    # 7860 since huggingface spaces demands this port
    uvicorn server:app --host 0.0.0.0 --port 7860
fi
