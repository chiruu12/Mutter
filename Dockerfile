FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY server/ server/
COPY client/__init__.py client/__init__.py
COPY client/cli.py client/cli.py
COPY pyproject.toml .

RUN pip install ".[docker]"

EXPOSE 7860

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]
