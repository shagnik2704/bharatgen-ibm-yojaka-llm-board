FROM python:3.11

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FRONTEND_RELATIVE_PATH="/frontend/index.html"

WORKDIR /app

# ✅ frontend is now visible
COPY frontend /frontend

# Install deps
COPY backend/requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# Copy backend code
COPY backend /app

EXPOSE 8005
CMD ["python", "app.py"]
