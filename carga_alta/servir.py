import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd

from . import config as C
from . import modelo as M

ESTADO = {}


def responder_previsao(corpo):
    cestos = corpo.get('cestos', [corpo]) if isinstance(corpo, dict) else corpo
    if not isinstance(cestos, list) or not cestos or not all(isinstance(c, dict) for c in cestos):
        raise ValueError('Envie um objeto com os sinais de um cesto ou {"cestos": [ {...}, {...} ]}.')
    X = pd.DataFrame(cestos)
    desconhecidas = sorted(set(X.columns) - set(C.VARIAVEIS) - {'cd_corrida', 'nu_carregamento'})
    faltando = [v for v in C.VARIAVEIS if v not in X]
    modelo = ESTADO['modelo']
    notas = M.nota_de_risco(modelo, X)
    niveis = M.nivel(notas)
    motivos = M.motivos(modelo, X)
    resultados = []
    for i, c in enumerate(cestos):
        r = {k: c[k] for k in ('cd_corrida', 'nu_carregamento') if k in c}
        r.update(nota_risco=float(notas[i]), nivel=str(niveis[i]), acao=C.ACOES[str(niveis[i])], motivos=motivos[i])
        resultados.append(r)
    avisos = []
    if faltando:
        avisos.append(f'{len(faltando)} sinais não enviados foram preenchidos pela mediana do treino: {", ".join(faltando)}')
    if desconhecidas:
        avisos.append(f'Campos ignorados: {", ".join(desconhecidas)}')
    return dict(resultados=resultados, avisos=avisos)


class Handler(BaseHTTPRequestHandler):
    server_version = 'CargaAlta/1.0'

    def _json(self, status, obj):
        dados = json.dumps(obj, ensure_ascii=False, indent=2).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(dados)))
        self.end_headers()
        self.wfile.write(dados)

    def do_GET(self):
        if self.path in ('/', '/saude'):
            meta = ESTADO['meta']
            self._json(200, dict(status='ok', modelo=meta.get('modelo'), treinado_em=meta.get('treinado_em'),
                                 dados_de_treino=meta.get('dados_de_treino'),
                                 niveis=dict(verde=f'nota < {C.CORTE_AMARELO}',
                                             amarelo=f'{C.CORTE_AMARELO} <= nota < {C.CORTE_VERMELHO}',
                                             vermelho=f'nota >= {C.CORTE_VERMELHO}'),
                                 rotas=['GET /saude', 'GET /variaveis', 'POST /prever']))
        elif self.path == '/variaveis':
            self._json(200, C.DESCRICAO_VARIAVEIS)
        else:
            self._json(404, dict(erro=f'Rota {self.path} não existe. Use /saude, /variaveis ou POST /prever.'))

    def do_POST(self):
        if self.path != '/prever':
            self._json(404, dict(erro='Use POST /prever.'))
            return
        try:
            tamanho = int(self.headers.get('Content-Length', 0))
            corpo = json.loads(self.rfile.read(tamanho).decode('utf-8') or 'null')
        except (ValueError, UnicodeDecodeError):
            self._json(400, dict(erro='O corpo da requisição não é um JSON válido.'))
            return
        try:
            self._json(200, responder_previsao(corpo))
        except ValueError as e:
            self._json(400, dict(erro=str(e)))

    def log_message(self, fmt, *args):
        print(f'[servir] {self.address_string()} {fmt % args}', flush=True)


def executar(porta=8000):
    if not C.ARQ_MODELO.exists():
        print('[servir] nenhum modelo encontrado: rodando preparar e treinar antes de subir a API')
        from . import preparar, treinar
        preparar.executar()
        treinar.executar()
    ESTADO['modelo'], ESTADO['meta'] = M.carregar()
    servidor = ThreadingHTTPServer(('0.0.0.0', porta), Handler)
    print(f'[servir] API no ar em http://localhost:{porta}  (Ctrl+C para parar)', flush=True)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        servidor.server_close()
