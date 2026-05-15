#!/usr/bin/env bash
set -euo pipefail

ML_URL="${ML_URL:-http://localhost:9090}"

echo "== image classification prediction =="
curl -sS -X POST "${ML_URL}/predict" \
  -H 'Content-Type: application/json' \
  -d '{
    "tasks": [
      {
        "id": "img-1",
        "data": {
          "image": "/data/local-files/?d=images/demo_blue.png",
          "caption": "blue sample with an object"
        }
      }
    ],
    "label_config": "<View><Image name=\"image\" value=\"$image\"/><Choices name=\"image_label\" toName=\"image\"><Choice value=\"Product\"/><Choice value=\"Other\"/></Choices></View>"
  }'

echo
echo "== image detection prediction =="
curl -sS -X POST "${ML_URL}/predict" \
  -H 'Content-Type: application/json' \
  -d '{
    "tasks": [
      {
        "id": "img-1",
        "data": {
          "image": "/data/local-files/?d=images/demo_blue.png"
        }
      }
    ],
    "label_config": "<View><Image name=\"image\" value=\"$image\"/><RectangleLabels name=\"bbox_label\" toName=\"image\"><Label value=\"Object\"/></RectangleLabels></View>"
  }'

echo
echo "== text classification prediction =="
curl -sS -X POST "${ML_URL}/predict" \
  -H 'Content-Type: application/json' \
  -d '{
    "tasks": [
      {
        "id": "txt-1",
        "data": {
          "text": "OpenAI released a helpful demo and I love it."
        }
      }
    ],
    "label_config": "<View><Text name=\"text\" value=\"$text\"/><Choices name=\"text_label\" toName=\"text\"><Choice value=\"Positive\"/><Choice value=\"Negative\"/></Choices></View>"
  }'

echo
echo "== text NER prediction =="
curl -sS -X POST "${ML_URL}/predict" \
  -H 'Content-Type: application/json' \
  -d '{
    "tasks": [
      {
        "id": "txt-2",
        "data": {
          "text": "Alice traveled from Paris to Berlin for OpenAI."
        }
      }
    ],
    "label_config": "<View><Text name=\"text\" value=\"$text\"/><Labels name=\"ner_label\" toName=\"text\"><Label value=\"ORG\"/><Label value=\"PERSON\"/><Label value=\"LOC\"/></Labels></View>"
  }'

echo
