FROM python:3.11-slim

WORKDIR /app
COPY app.py /app/app.py
COPY segmentation_corrections.py /app/segmentation_corrections.py
COPY segmentation_correction_risk.py /app/segmentation_correction_risk.py
COPY segmentation_review_queue.py /app/segmentation_review_queue.py

RUN pip install --no-cache-dir numpy scikit-learn==1.5.2 pillow==10.4.0 label-studio-converter==0.0.59

ENV PYTHONUNBUFFERED=1
CMD ["python", "/app/app.py"]
