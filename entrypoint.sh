#!/bin/sh
set -e
# Fail on any error


if [ "$ENV_TYPE" = "dev" ]; then
    echo "[ENV: DEV] Starting FastAPI (8000) + Streamlit (8501) ..."
    echo "Make sure to map both 8000 and 8501 ports in your Docker run/create command."

    # Start FastAPI server on 8000:
    cd /fastAPI
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload &

    # Start Streamlit app on 8501:
    cd /streamlit
    streamlit run app.py --server.port 8501

else
    echo "[ENV: PROD] Starting FastAPI (8000) + Streamlit (7860) ..."
    echo "Only Streamlit on 7860 will be publicly accessible due to Hugging Face Spaces limitations."

    # Start FastAPI server on 8000:
    cd /fastAPI
    uvicorn server:app --host 0.0.0.0 --port 8000 &

    # Start Streamlit app on 7860:
    cd /streamlit
    streamlit run app.py --server.port 7860 --server.headless true

fi
