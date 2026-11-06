set -e
cd "$(dirname "$0")"
docker build -t carga-alta:1.0 .
docker run --rm -v "$(pwd)/saida:/app/saida" carga-alta:1.0 tudo
echo "Pronto. Veja saida/avaliacao/relatorio.md e saida/previsoes/previsoes.csv"
