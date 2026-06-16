FROM python:3.11-slim

# Dépendances système pour FAISS (CPU) et BeautifulSoup
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier les requirements en premier (cache Docker)
COPY requirements.txt .

# Installer les modules qui sont dans requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Copier le code applicatif
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
