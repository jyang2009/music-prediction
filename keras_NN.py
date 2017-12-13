from keras.models import Model
from keras.layers import Input, Dense
from keras.models import Sequential
from keras.layers import Activation
from keras.callbacks import EarlyStopping
from keras import backend as K
import tensorflow as tf
import math
sess = tf.Session()
K.set_session(sess)

import numpy as np
B=np.genfromtxt('map_gaussian_mixture1.txt',delimiter=',')
ds = tf.contrib.distributions
mix = list(B[:,-1])
sigma=1.
myc=[ds.MultivariateNormalDiag([B[0,0],B[0,1]], [B[0,2]/sigma,B[0,2]/sigma])]

for i in range(59):
    myc.append(ds.MultivariateNormalDiag([B[i+1,0],B[i+1,1]], [B[i+1,2]/sigma,B[i+1,2]/sigma]))
    
bimix_gauss = ds.Mixture(
  cat=ds.Categorical(probs=mix),
  components=myc)

def _loss_tensor(y_true, y_pred):
    split0, split1 = tf.split(y_pred, num_or_size_splits=2, axis=-1)
    split3, split4 = tf.split(y_true, num_or_size_splits=2, axis=-1)
    out=2*6371.*tf.asin(tf.sqrt(K.sin((split1 - split4)/(180*math.pi))**2+K.cos(split1/(180*math.pi))*K.cos(split4/(180*math.pi))*K.sin((split3-split0)*0.5/(180*math.pi))**2))
    out2=-1e4*bimix_gauss.prob(y_pred)
    return K.mean(out, axis=-1)+K.mean(out2, axis=-1)

model = Sequential([
    Dense(50, input_shape=(68,)),
    Activation('relu'),
    Dense(50),
    Activation('relu'),
    Dense(2),
    Activation('linear'),
])



model.compile(optimizer='rmsprop',
              loss= _loss_tensor)
import numpy as np

A=np.genfromtxt('train0.8.txt')
C=np.genfromtxt('test0.2.txt')


N=np.shape(A)[0]
m=np.shape(A)[1]

X=A[:,0:-2]
Y=A[:,-2:]
X_v=C[:,0:-2]
Y_v=C[:,-2:]
# Train the model, iterating on the data in batches
early_stopping = EarlyStopping(monitor='val_loss', patience= 20)
model.fit(X, Y, epochs = 100, batch_size = 50, validation_data = [X_v, Y_v], callbacks=[early_stopping])
#model.fit(X, Y, epochs=100, batch_size=50,validation_data=[X_v,Y_v],verbose=2)
Y_p=model.predict(X_v)

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
bm=Basemap()
bm.drawcoastlines(linewidth=0.5)
bm.drawmeridians(np.arange(0,360,30))
bm.drawparallels(np.arange(-90,90,30))
#Y_p=predict(X,m_l,W,b)
plt.scatter(Y_v[:,1],Y_v[:,0])
plt.scatter(Y_p[:,1],Y_p[:,0])
