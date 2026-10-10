# ── Imagem base ──────────────────────────────────────────────────────────────
# python:3.11-slim é leve (~130MB) e suficiente para APIs de ML
FROM python:3.11-slim

# ── Metadados ─────────────────────────────────────────────────────────────────
LABEL maintainer="Rafael Simões <rafael.fnx07@gmail.com>"
LABEL description="DeliveryRisk API — previsão de atraso de entregas com Random Forest"
LABEL version="1.0.0"

# ── Variáveis de ambiente ─────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=model/model.pkl \
    PORT=8080

# ── Diretório de trabalho ─────────────────────────────────────────────────────
WORKDIR /app

# ── Instalar dependências ─────────────────────────────────────────────────────
# Copia requirements primeiro para aproveitar o cache do Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copiar código e modelo ────────────────────────────────────────────────────
COPY app/ ./app/
COPY model/ ./model/

# ── Porta exposta (Cloud Run usa 8080 por padrão) ─────────────────────────────
EXPOSE 8080

# ── Comando de inicialização ──────────────────────────────────────────────────
# --host 0.0.0.0 necessário para o Cloud Run receber tráfego externo
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
