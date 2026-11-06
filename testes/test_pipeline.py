import importlib.util
import os
import shutil
import tempfile
import unittest

_TMP = tempfile.mkdtemp(prefix='carga_alta_teste_')
os.environ['SAIDA'] = _TMP 

import pandas as pd  

from carga_alta import avaliar, config as C, prever, preparar, servir, treinar 
from carga_alta import modelo as M  


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cestos = preparar.executar()
        treinar.executar()
        cls.comp = avaliar.executar()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(_TMP, ignore_errors=True)

    def test_dados_por_cesto(self):
        t = self.cestos
        self.assertEqual(len(t), 3501)
        self.assertEqual(t.cd_corrida.nunique(), 1799)
        self.assertTrue(set(C.VARIAVEIS) <= set(t.columns))
        self.assertFalse(t.duplicated(['cd_corrida', 'nu_carregamento']).any())

    def test_historico_nao_olha_o_futuro(self):
      
        primeira = self.cestos.sort_values('ordem').groupby('nu_carregamento').head(1)
        self.assertTrue(primeira.y_ant.isna().all())
        self.assertTrue(primeira.taxa_ult20.isna().all())

    def test_resultado_da_apresentacao(self):
        m = self.comp[self.comp.avaliacao == 'mês de prova'].set_index('alternativa')
        sol = m.loc['Votação suave (solução)']
        self.assertEqual(int(sol.alertas), 59)
        self.assertEqual(int(sol.paradas), 57)
        self.assertEqual(int(sol.paradas_antecipadas), 38)
        self.assertEqual(int(m.loc['Modelo anterior (regressão logística)', 'paradas_antecipadas']), 34)
        self.assertEqual(int(m.loc['Regra simples (corrida anterior parou)', 'paradas_antecipadas']), 23)
        self.assertAlmostEqual(sol.auc, 0.887, places=3)

    def test_previsao_e_niveis(self):
        out = prever.executar()
        self.assertEqual(len(out), 352)
        self.assertTrue(out.nota_risco.between(0, 100).all())
        self.assertTrue(set(out.nivel) <= {'verde', 'amarelo', 'vermelho'})
        verm = out[out.nivel == 'vermelho']
        self.assertGreater(verm[C.ALVO].mean(), out[C.ALVO].mean() * 2) 


    def test_api(self):
        servir.ESTADO['modelo'], servir.ESTADO['meta'] = M.carregar()
        linha = self.cestos[C.VARIAVEIS].iloc[0].to_dict()
        r = servir.responder_previsao({'cestos': [linha, linha]})
        self.assertEqual(len(r['resultados']), 2)
        self.assertIn(r['resultados'][0]['nivel'], C.ACOES)
        with self.assertRaises(ValueError):
            servir.responder_previsao('texto')


if __name__ == '__main__':
    unittest.main()


@unittest.skipUnless(importlib.util.find_spec('streamlit'), 'streamlit não instalado')
class TestApp(unittest.TestCase):
    def test_paginas(self):
        from streamlit.testing.v1 import AppTest
        app = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app_streamlit.py')
        at = AppTest.from_file(app, default_timeout=180)
        at.run()
        self.assertFalse(at.exception, at.exception)
        for pagina in ['Simular um cesto', 'Mês de prova', 'Prever arquivo CSV', 'Sobre o projeto', 'Início']:
            at.sidebar.radio[0].set_value(pagina).run()
            self.assertFalse(at.exception, f'{pagina}: {at.exception}')
