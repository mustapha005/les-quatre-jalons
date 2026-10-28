import numpy as np
import tensorflow as tf
from tensorflow import keras
SEED=42
CLASSES = ['T-shirt/top', 'Pantalon', 'Pull', 'Robe', 'Manteau', 'Sandale',
           'Chemise', 'Basket', 'Sac', 'Bottine']
def session_features(s,split,rich=False):
    typ=s['type_'+split].astype(int);cat=s['categorie_'+split].astype(int);length=s['longueur_'+split].astype(int)
    valid=np.arange(60)[None,:]<length[:,None];assert np.array_equal(valid,typ!=0)
    base=np.stack([(typ==k).sum(1) for k in range(1,6)],axis=1).astype('float32')
    if not rich:return base
    duration=s['duree_'+split].astype('float32');price=s['prix_'+split].astype('float32')
    category_counts=np.stack([(cat==k).sum(1) for k in range(1,11)],axis=1)
    totals=np.stack([length,duration.sum(1),duration.sum(1)/length,(price*valid).sum(1)/length],axis=1)
    last=[];row=np.arange(len(length))
    for j in range(5):
        present=length>j;idx=np.maximum(length-1-j,0)
        t=typ[row,idx];last.append(np.stack([(t==k)&present for k in range(1,6)],axis=1))
        last.append(np.stack([np.where(present,duration[row,idx],0),np.where(present,price[row,idx],0),present],axis=1))
    return np.concatenate([base,category_counts,totals,*last],axis=1).astype('float32')

@keras.utils.register_keras_serializable(package='UC2')
class SmartphoneBrightness(keras.layers.Layer):
    """Augmentation entraînement seulement, bornée dans [0,1]."""
    def __init__(self,strength=.12,seed=SEED,**kwargs):
        super().__init__(**kwargs);self.strength=strength;self.seed=seed;self.generator=keras.random.SeedGenerator(seed)
    def call(self,x,training=None):
        if training:
            shift=keras.random.uniform((keras.ops.shape(x)[0],1,1,1),minval=-self.strength,maxval=self.strength,seed=self.generator)
            return keras.ops.clip(x+shift,0.,1.)
        return x
    def get_config(self):return {**super().get_config(),'strength':self.strength,'seed':self.seed}
