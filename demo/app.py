"""Démonstration locale UC2 : validation seulement, aucune formation."""
from pathlib import Path
import sys, json
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd
import joblib
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from inference import keras,CLASSES,session_features,SmartphoneBrightness
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(page_title='AtlasShop · UC2',layout='wide')
st.title('AtlasShop — démonstration des quatre jalons')
st.caption('Groupe 13 · Mustapha Aarab · Aya Agrigah · Toubani Badr Eddine')
st.caption('Données pédagogiques de validation · modèles sauvegardés · aucune action commerciale')

@st.cache_data
def data(name):
    with np.load(ROOT/'data_UC2'/name,allow_pickle=False) as z:
        return {k:z[k] for k in z.files if k.endswith('_val')}

@st.cache_resource
def model(j,name):
    return keras.models.load_model(ROOT/f'artifacts/jalon{j}/models/{name}/best.keras',compile=False)

@st.cache_resource
def saved(j,name):return joblib.load(ROOT/f'artifacts/jalon{j}'/name)

def artifact(j,name):return json.loads((ROOT/f'artifacts/jalon{j}'/name).read_text(encoding='utf-8'))
def bow(j,text,filename='vocabulary.json'):
    vector=CountVectorizer(binary=True,vocabulary=artifact(j,filename))
    return vector.transform([text]).toarray().astype('float32')
def infer(j,name,x):
    value=model(j,name)(x,training=False)
    return {k:v.numpy() for k,v in value.items()} if isinstance(value,dict) else value.numpy()
def show_class(prob,true=None):
    k=int(np.argmax(prob));st.write('Catégorie prédite : **'+CLASSES[k]+'**')
    if true is not None:st.write('Étiquette validation : '+CLASSES[int(true)])
    st.bar_chart(pd.DataFrame({'score':prob.ravel()},index=CLASSES))

def display_gray(pixels, caption):
    """Embed a grayscale PNG directly in the page."""
    arr = np.asarray(pixels)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.rint(np.clip(arr, 0, 1) * 255)
    img = Image.fromarray(arr.astype(np.uint8)).resize((224, 224), Image.Resampling.NEAREST)
    buf = BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    st.html(f'<figure style="margin:0"><img width="224" height="224" alt="{caption}" src="data:image/png;base64,{encoded}"><figcaption>{caption}</figcaption></figure>')

page=st.sidebar.selectbox('Jalon',['1 · MLP et prix','2 · CNN et auto-encodeur','3 · Boutique B et multitâche','4 · Sessions GRU'])
if page.startswith(('1','2','3')):
    j=int(page[0]);d=data('uc2_catalogueB.npz' if j==3 else 'uc2_catalogueA.npz')
    index=st.slider('Annonce de validation',0,len(d['X_val'])-1,0)
    image=d['X_val'][index];display_gray(image,'Annonce de validation')
    text=st.text_input('Description (modifiable)',str(d['description_val'][index]))
    st.caption('Modifier la description produit une prédiction exploratoire, sans nouvelle évaluation de performance.')
    x=image[None,...,None].astype('float32')/255.;words=bow(j,text)
    if j==1:
        inp={'image':x.reshape(1,784),'mots':words}
        show_class(infer(1,'MLP_reference',inp),d['categorie_val'][index])
        upper=json.loads((ROOT/'data_UC2/metadata.json').read_text())['uc2_catalogueA.npz']['price_clip_upper']
        estimate=float(np.exp(np.clip(infer(1,'price_log_huber',inp).ravel()[0],0,np.log(upper))))
        st.metric('Prix proposé (DH)',f'{estimate:.2f}')
        st.write('Prix fourni validation :',round(float(d['prix_val'][index]),2),'DH')
        st.subheader('Troisième tâche : achat par MLP')
        sessions=data('uc2_sessions.npz')
        si=st.slider('Session pour le MLP',0,len(sessions['achat_val'])-1,0)
        row={k:v[si:si+1] for k,v in sessions.items()}
        features=session_features(row,'val',rich=True)
        features=saved(1,'session_rich_mlp_scaler.joblib').transform(features).astype('float32')
        score=float(infer(1,'session_rich_mlp',features).ravel()[0])
        st.metric('Score d’achat MLP',f'{score:.3f}')
        st.caption(f"Étiquette de validation : {int(row['achat_val'][0])}. Score calculé sur la session complète.")
    elif j==2:
        show_class(infer(2,'CNN_base',{'image':x,'mots':words}),d['categorie_val'][index])
        st.write('Auto-encodeur : choix sur proxy val, seuil sur images val ID')
        probe=st.selectbox('Image à reconstruire',['Annonce ci-dessus','Rectangle synthétique de démonstration','Bruit synthétique de démonstration'])
        if probe.startswith('Rectangle'):
            x=np.zeros((1,28,28,1),dtype='float32');x[:,9:19,8:20,:]=.4
        elif probe.startswith('Bruit'):x=np.random.default_rng(42).random((1,28,28,1),dtype=np.float32)
        recon=infer(2,'AE_128',x);score=float(np.mean((x-recon)**2))
        threshold=artifact(2,'frozen_protocol.json')['selection']['threshold']
        left, right = st.columns(2)
        with left: display_gray(x[0,...,0], 'Original')
        with right: display_gray(recon[0,...,0], 'Reconstruction')
        st.write(f'MSE={score:.6f} ; seuil={threshold:.6f} ; '+('rejet' if score>threshold else 'acceptation'))
        st.caption('Une acceptation ne certifie pas une annonce valide. Les proxies du démonstrateur ne constituent pas un score OOD officiel.')
    else:
        standard=artifact(3,'log_price_scaler.json')
        out=infer(3,'MT_warm_lambda_1p0',{'image':x,'mots':words})
        show_class(out['categorie'],d['categorie_val'][index])
        logprice=float(out['logprix'].ravel()[0])*standard['std']+standard['mean']
        upper=json.loads((ROOT/'data_UC2/metadata.json').read_text())['uc2_catalogueB.npz']['price_clip_upper']
        estimate=float(np.exp(np.clip(logprice,np.log(1),np.log(upper))))
        st.metric('Prix multitâche (DH)',f'{estimate:.2f}')
        st.write('Prix fourni validation :',round(float(d['prix_val'][index]),2),'DH')
        st.caption('Modèle réservé à B. L’adaptation dégrade la performance sur A : le CNN A reste versionné séparément.')
else:
    d=data('uc2_sessions.npz');index=st.slider('Session de validation',0,len(d['achat_val'])-1,0)
    length=int(d['longueur_val'][index]);types=d['type_val'][index:index+1].astype('int32');cats=d['categorie_val'][index:index+1].astype('int32')
    numeric=np.stack([d['prix_val'][index:index+1].astype('float32')/5,d['duree_val'][index:index+1].astype('float32')],axis=-1)
    numeric=saved(4,'event_numeric_scaler.joblib').transform(numeric.reshape(-1,2)).reshape(1,60,2).astype('float32');numeric[types==0]=0
    fields={'type':types,'categorie':cats,'numerique':numeric}
    name=artifact(4,'test_results.json')['purchase']['model']
    probability=float(infer(4,name,fields).ravel()[0]);threshold=artifact(4,'frozen_protocol.json')['selection']['coupon_threshold_val']
    st.metric('Score d’achat de la session complète',f'{probability:.3f}')
    st.write(f'{length} événements ; étiquette validation : {int(d["achat_val"][index])} ; seuil val complet : {threshold:.6f}')
    st.write('Au-dessus du seuil numérique :',probability>=threshold)
    st.dataframe(pd.DataFrame({'type':types[0,:length],'categorie':cats[0,:length],
        'gamme_prix':d['prix_val'][index,:length],'duree':d['duree_val'][index,:length]}),hide_index=True)
    st.caption('Pour un lot, le budget exact est le top 10% par rang. Un seuil fixe ne garantit pas cette proportion. Ici le score utilise une session complète, sans promesse de ciblage en ligne. Sous cinq événements : coupon différé.')
    st.warning('Le GRU retenu sans masque dépend du padding fixe à 60. Les variantes masquées sont disponibles dans le notebook ; ce démonstrateur conserve le choix validation du run livré.')
    st.subheader('Description : modèle retenu après comparaison de l’ordre des mots')
    description=st.text_input('Description à classifier','robe noire élégante')
    show_class(infer(4,'BoW_MLP',bow(4,description,'bow_vocabulary.json')))
    st.caption('Le BoW a remporté la sélection validation face aux LSTM de texte. Les expériences séquentielles complètes figurent dans le notebook du jalon 4.')

with st.expander('Résultats enregistrés du jalon et limites'):
    st.json(artifact(int(page[0]),'test_results.json'))
    st.caption('Résultats du run des notebooks, sur les partitions de test complètes. Ils ne sont pas recalculés sur les 200 exemples de validation inclus dans cette application. Prix, descriptions et sessions pédagogiques simulés ; aucune garantie de performance sur une vraie marketplace.')
