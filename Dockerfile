FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DADOS_BRUTOS=/app/dados/brutos \
    SAIDA=/app/saida

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY carga_alta/ carga_alta/
COPY testes/ testes/
COPY dados/brutos/ dados/brutos/
COPY exemplos/ exemplos/

VOLUME ["/app/saida"]
EXPOSE 8000

ENTRYPOINT ["python", "-m", "carga_alta"]

CMD ["tudo"]
