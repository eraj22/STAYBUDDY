# Pin the verified base so a later chatbot rebuild does not silently change Python.
FROM python:3.13-slim@sha256:8d9d0b8bcf6506481eae4907c18f5e3e7902e629f5f6d684f9e7c32e85e3ddf0
WORKDIR /services
COPY ml-notebooks/requirements.txt /tmp/recommendation.txt
COPY ml-notebooks/ml_ai_complaint/requirements.txt /tmp/complaint.txt
COPY ml-notebooks/room_matcher/requirements.txt /tmp/roommate.txt
RUN pip install --no-cache-dir -r /tmp/recommendation.txt -r /tmp/complaint.txt -r /tmp/roommate.txt
COPY ml-notebooks /services
