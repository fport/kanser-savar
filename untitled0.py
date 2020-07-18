import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score  #gerçek verilerl örtüştüğünü anlamak
import pandas as pd
from sklearn.model_selection import train_test_split 



veri = pd.read_csv("breast-cancer-wisconsin.data")

veri.replace('?', -99999, inplace=True) # ? olan veriler var datamızda onları cıkardık
veri = veri.drop(['id'], axis=1) #axis sütün kolon yukardan asagı dogru olan id kısmını sılme


y = np.array(veri.benormal)
x = np.array(veri.drop(['benormal'], axis=1))

imp = Imputer(missing_values=-99999, strategy="mean",axis=0)
x = imp.fit_transform(x)

"""
for z in range(25):
    z = 2*z+1
    print("En yakın",z,"komşu kullandığımızda tutarlılık oranımız")
    tahmin = KNeighborsClassifier(n_neighbors=z, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='euclidean', metric_params=None, n_jobs=1)
    tahmin.fit(x,y)
    ytahmin = tahmin.predict(x)

    basari = accuracy_score(y, ytahmin, normalize=True, sample_weight=None)
    print(basari)
    """
    

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.33)

tahmin = KNeighborsClassifier()
tahmin.fit(X_train,y_train)


basaria = tahmin.score(X_test, y_test) 
print("Yüzde",basaria*100," oranında:" )

a = np.array([[1,5,2,2,3,2,1,2,3]])
a.reshape(1,-1)
b = tahmin.predict(a)
if(b == 2):
    print("benign tumor")
else:
    print("malignant tumor")
#2 iyi huylu kanser 4 kötü