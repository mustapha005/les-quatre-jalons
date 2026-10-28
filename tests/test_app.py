"""Smoke tests des prédictions réelles, sans entraînement."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest
root=Path(__file__).resolve().parents[1]
a=AppTest.from_file(str(root/'streamlit_app.py'),default_timeout=120).run()
def check(label):
    assert not a.exception, [(x.message) for x in a.exception]
    print('PASS',label,flush=True)
check('J1 catégorie, prix, achat')
a.slider[0].set_value(199).run();check('J1 dernier exemple')
for page in a.sidebar.selectbox[0].options[1:]:
    a.sidebar.selectbox[0].select(page).run();check(page)
    a.slider[0].set_value(199).run();check(page+' dernier exemple')
    if page.startswith('2'):
        for probe in a.selectbox[0].options if a.selectbox[0].label=='Image à reconstruire' else a.selectbox[1].options:
            control=next(s for s in a.selectbox if s.label=='Image à reconstruire')
            control.select(probe).run();check(probe)
    if a.text_input:
        a.text_input[0].set_value('').run();check(page+' texte vide')
print('Toutes les vues passent sans exception.')
