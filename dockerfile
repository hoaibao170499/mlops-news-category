FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# build normal model & copy to this dockerfile (not recommended)

RUN pip install -r requirements.txt

COPY scripts/ scripts/

ENV PYTHONPATH=/app

RUN python -m nltk.downloader stopwords words wordnet punkt punkt_tab

CMD ["python3", "scripts/api.py"]