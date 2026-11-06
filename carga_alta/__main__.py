import argparse
import time


def main(argv=None):
    p = argparse.ArgumentParser(prog='carga_alta', description='Pipeline da nota de risco de carga alta (EVCOMX).')
    sub = p.add_subparsers(dest='etapa', required=True)
    sub.add_parser('tudo', help='roda preparar, treinar, avaliar e prever em sequência')
    sub.add_parser('preparar', help='gera a tabela por cesto a partir dos arquivos brutos')
    t = sub.add_parser('treinar', help='treina e salva o modelo')
    t.add_argument('--todos', action='store_true', help='treina com todas as corridas (retreino mensal)')
    sub.add_parser('avaliar', help='avalia com o protocolo do projeto')
    pr = sub.add_parser('prever', help='dá a nota de risco de cada cesto de um CSV')
    pr.add_argument('--entrada', help='CSV com uma linha por cesto e os 27 sinais (padrão: cestos do mês de prova)')
    pr.add_argument('--saida', help='CSV de saída (padrão: SAIDA/previsoes/previsoes.csv)')
    s = sub.add_parser('servir', help='sobe a API HTTP')
    s.add_argument('--porta', type=int, default=8000)
    ap = sub.add_parser('app', help='sobe o app Streamlit')
    ap.add_argument('--porta', type=int, default=8501)
    a = p.parse_args(argv)

    if a.etapa == 'app':
        import subprocess
        import sys
        from pathlib import Path
        app = Path(__file__).resolve().parent.parent / 'app_streamlit.py'
        cmd = [sys.executable, '-m', 'streamlit', 'run', str(app), '--server.address', '0.0.0.0',
               '--server.port', str(a.porta), '--server.headless', 'true', '--browser.gatherUsageStats', 'false']
        print(f'[app] app Streamlit em http://localhost:{a.porta}  (Ctrl+C para parar)', flush=True)
        raise SystemExit(subprocess.call(cmd, cwd=app.parent))

    inicio = time.time()
    if a.etapa in ('tudo', 'preparar'):
        from . import preparar
        preparar.executar()
    if a.etapa in ('tudo', 'treinar'):
        from . import treinar
        treinar.executar(todos=getattr(a, 'todos', False))
    if a.etapa in ('tudo', 'avaliar'):
        from . import avaliar
        avaliar.executar()
    if a.etapa in ('tudo', 'prever'):
        from . import prever
        prever.executar(entrada=getattr(a, 'entrada', None), saida=getattr(a, 'saida', None))
    if a.etapa == 'servir':
        from . import servir
        servir.executar(porta=a.porta)
    print(f'[{a.etapa}] concluído em {time.time() - inicio:.0f} s')


def rodando_no_streamlit():
    try:
        from streamlit import runtime
        return runtime.exists()
    except Exception:
        return False


if __name__ == '__main__':
    if rodando_no_streamlit():
        # "streamlit run carga_alta/__main__.py" (ex.: Streamlit Community Cloud) abre o app
        import runpy
        import sys
        from pathlib import Path
        raiz = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(raiz))
        runpy.run_path(str(raiz / 'app_streamlit.py'), run_name='__main__')
    else:
        main()
