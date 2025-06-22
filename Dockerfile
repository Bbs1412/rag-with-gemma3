FROM python:3.12-slim-bookworm

# Install dependencies
# RUN apt update && apt upgrade -y && rm -rf /var/lib/apt/lists/*
# RUN apt update && apt install -y --no-install-recommends curl ca-certificates && \
#     rm -rf /var/lib/apt/lists/*

# Copy the application files
COPY server/ /fastAPI/
COPY assets/ /streamlit/assets/
COPY app.py /streamlit/app.py
COPY .streamlit/ /streamlit/.streamlit/
COPY server/logger.py /streamlit/logger.py

# Copy the files to replace:
COPY docker/. /temp_files/
COPY requirements_dev.txt /temp_files/requirements_dev.txt
COPY requirements_deploy.txt /temp_files/requirements_deploy.txt

# Get the build arguments, with a default values
ARG ENV_TYPE=dev
# Set the environment variable in the container
ENV ENV_TYPE=${ENV_TYPE}


# Check the env and replace the files accordingly
RUN if [ "$ENV_TYPE" = "deploy" ]; then \
        # Copy the deployment files
        cp /temp_files/deploy_llm.py /fastAPI/llm_system/core/llm.py && \
        cp /temp_files/deploy_database.py /fastAPI/llm_system/core/database.py && \
        cp /temp_files/requirements_deploy.txt /fastAPI/requirements.txt; \
    else \
        # Copy the development files 
        # (it is not from Docker Engine, it is inside the container to container only)
        cp /temp_files/dev_llm.py /fastAPI/llm_system/core/llm.py && \
        cp /temp_files/dev_database.py /fastAPI/llm_system/core/database.py && \
        cp /temp_files/requirements_dev.txt /fastAPI/requirements.txt; \
    fi


# delete temporary files
WORKDIR /fastAPI
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt
RUN rm -rf /temp_files/ && pip cache purge

# EXPOSE PORTS
# dev -> EXPOSE 8000, 8501, 11434
# deploy -> EXPOSE 7860

# Move towards the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
CMD ["/entrypoint.sh"]
