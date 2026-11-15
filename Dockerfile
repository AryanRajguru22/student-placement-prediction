# Optional container image for the Student Placement Probability Predictor (the primary deployment target is Streamlit Community Cloud).
#   docker build -t placement-predictor .   &&   docker run --rm -p 8501:8501 placement-predictor
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv
COPY app/requirements.txt app/requirements.txt
RUN pip install --disable-pip-version-check -r app/requirements.txt
# only what the app needs at run time: code, config and the frozen model (verified by SHA-256 on start-up); no data, notebooks or reports
COPY .streamlit .streamlit
COPY app/*.py app/model_metadata.json app/
COPY models/phase7/LR_B_final.joblib models/phase7/LR_B_final.joblib
RUN useradd --create-home appuser && chown -R appuser /srv
USER appuser
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)"
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
