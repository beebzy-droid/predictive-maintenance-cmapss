FROM python:3.11-slim

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY --chown=user requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of the app
COPY --chown=user . /app

# Streamlit runs on port 8501 by default; HF Spaces routes traffic here
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]