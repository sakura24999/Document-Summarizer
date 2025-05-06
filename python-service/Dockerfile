FROM python:3.10-slim

# システム依存関係のインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依存関係のインストール
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=600 -r requirements.txt
RUN pip install --no-cache-dir httpx==0.24.1

# アプリケーションコードのコピー
COPY ./app /app

# FastAPI アプリケーションの実行
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

EXPOSE 8000
