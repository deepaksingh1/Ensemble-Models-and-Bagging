import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix

import warnings
warnings.filterwarnings('ignore')
df = pd.read_excel('BreastTissue.xlsx', sheetname='Data',index_col=0)

X = df.drop(['Class'], axis=1)
y = df['Class']

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

#========LogisticRegression======================
from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression(random_state=0)
clf_lr.fit(X_train,y_train)
y_predlr = clf_lr.predict(X_test)
acc_lr = accuracy_score(y_test,y_predlr)
cm_lr = confusion_matrix(y_test,y_predlr)
print(acc_lr)
print(cm_lr)

#========Decision Tree===========================
from sklearn.tree import DecisionTreeClassifier
clf_dc = DecisionTreeClassifier(random_state=0)
clf_dc.fit(X_train,y_train)
y_preddc = clf_dc.predict(X_test)
acc_dc=accuracy_score(y_test,y_preddc)
cm_dc=confusion_matrix(y_test,y_preddc)
print(acc_dc)
print(cm_dc)

#======KNN=================================
from sklearn.neighbors import KNeighborsClassifier
clf_kn = KNeighborsClassifier(n_neighbors=3, metric = 'euclidean').fit(X_train, y_train)
ypred_kn = clf_kn.predict(X_test)
acc_kn=accuracy_score(y_test,ypred_kn)
cm_kn=confusion_matrix(y_test,ypred_kn)
print(acc_kn)
print(cm_kn)

#==============Bagging============================
from sklearn.ensemble import BaggingClassifier

bag_lr = BaggingClassifier(clf_lr, random_state=2, max_samples=0.5, max_features=0.5)
bag_lr.fit(X_train, y_train)
acc_blr = bag_lr.score(X_test, y_test)
print(acc_blr)

bag_dc = BaggingClassifier(clf_dc, random_state=2, max_samples=0.5, max_features=0.5)
bag_dc.fit(X_train, y_train)
acc_bdc = bag_dc.score(X_test, y_test)
print(acc_bdc)

bag_kn = BaggingClassifier(clf_kn,random_state=2,max_samples=0.5, max_features=0.5)
bag_kn.fit(X_train, y_train)
acc_bkn = bag_kn.score(X_test, y_test)
print(acc_bkn)

#===========Visulaization==========================
import pandas as pd
import seaborn as sns
#=============LR=================================
"""
x=['lr','blr']
y=[acc_lr*100,acc_cvlr.mean()*100,acc_blr*100,acc_cvblr.mean()*100]
cat=['first','second','third','fourth']
df = pd.DataFrame(dict(Algorithms=x, Accuracy=y,cat=cat))
ax1 = sns.barplot("Algorithms","Accuracy", data=df);
"""

x = ['lr','lr','dt','dt','KNN','KNN']
y=[acc_lr*100,acc_blr*100,
   acc_dc*100,acc_bdc*100,
   acc_kn*100,acc_bkn*100]
category = ['simple','cross_simple','bag','cross_bag',
       'simsimple','cross_simple','bag','cross_bag',
       'simple','cross_simple','bag','cross_bag']
df = pd.DataFrame(dict(Algorithms=x, Accuracy=y,cat=cat))
ax = sns.barplot(x="category", y="Accuracy", hue="Algorithms", data=df)
