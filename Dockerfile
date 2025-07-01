# Usa un'immagine Python ufficiale e leggera
FROM python:3.10-slim

# Imposta la directory di lavoro all'interno del container
WORKDIR /app

# Copia il file delle dipendenze e installale
# Questo passaggio viene fatto per primo per sfruttare la cache di Docker
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutto il resto del codice del progetto nella directory di lavoro
COPY . .

# Esponi la porta su cui il server Uvicorn sarà in ascolto
EXPOSE 8000

# Il comando per avviare il server quando il container parte
CMD ["uvicorn", "server.mcp_server:app", "--host", "0.0.0.0", "--port", "8000"]