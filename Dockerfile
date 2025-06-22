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
COPY requirements.txt /temp_files/requirements.txt

# Add entry point script
COPY entrypoint.sh /entrypoint.sh

# Create a non-root user and group
RUN useradd -m appuser
# Set permissions so appuser owns what it needs
RUN chmod +x /entrypoint.sh && \
    chown -R appuser:appuser /fastAPI /streamlit /temp_files /entrypoint.sh /tmp

# Get the build arguments, with a default values
ARG ENV_TYPE=deploy
# Set the environment variable in the container
ENV ENV_TYPE=${ENV_TYPE}


# Check the env and replace the files accordingly
RUN if [ "$ENV_TYPE" = "deploy" ]; then \
        # Copy the deployment files
        cp /temp_files/deploy_llm.py /fastAPI/llm_system/core/llm.py && \
        cp /temp_files/deploy_database.py /fastAPI/llm_system/core/database.py; \
    else \
        # Copy the development files 
        # (it is not from Docker Engine, it is inside the container to container only)
        cp /temp_files/dev_llm.py /fastAPI/llm_system/core/llm.py && \
        cp /temp_files/dev_database.py /fastAPI/llm_system/core/database.py; \
    fi


# Switch to non-root user
USER appuser
WORKDIR /temp_files
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Install dependencies:
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Switch back to root briefly to clean up temp files
WORKDIR /fastAPI
USER root
RUN rm -rf /temp_files && pip cache purge
USER appuser

# EXPOSE PORTS
# dev -> EXPOSE 8000, 8501, 11434
# deploy -> EXPOSE 7860

# Start entrypoint
CMD ["/entrypoint.sh"]