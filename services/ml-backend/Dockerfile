FROM python:3.11-slim

WORKDIR /app
COPY app.py /app/app.py
COPY segmentation_quality.py /app/segmentation_quality.py
COPY segmentation_uncertainty.py /app/segmentation_uncertainty.py

RUN pip install --no-cache-dir numpy scikit-learn==1.5.2 pillow==10.4.0 label-studio-converter==0.0.59

ENV PYTHONUNBUFFERED=1
CMD ["python", "/app/app.py"]
