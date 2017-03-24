from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np
import pandas as pd
from Titanic import preprocess as pp
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.model_selection import cross_val_score
print("modeling...")

train_fea=pp.fea[:891]
xcv_train,xcv_test,ycv_train,ycv_test=train_test_split(train_fea,pp.titanic.Survived,test_size=0.3)

#svm
svm=SVC(kernel='linear',C=1).fit(xcv_train,ycv_train)
print svm.score(xcv_test,ycv_test)
print metrics.roc_auc_score(ycv_test,svm.predict(xcv_test))
#random forest
sample_leaf_options = list(range(1, 500, 3))
n_estimators_options = list(range(1, 1000, 5))
parameters={'n_estimators':n_estimators_options,'max_features':('sqrt','')}
rf=RandomForestClassifier().fit(xcv_train,ycv_train)
print rf.score(xcv_test,ycv_test)
print metrics.roc_auc_score(ycv_test,rf.predict(xcv_test))

#
# x_train=pp.fea[:891]
# y_train=pp.full.Survived[:891]
# x_test=pp.fea[891:]
# model=SVC(kernel='linear',C=1).fit(pp.fea[:891],pp.titanic.Survived[:891])
# y_test=model.predict(x_test).astype(np.int)
#
# test_ids=pp.full[891:].PassengerId
# test=pd.DataFrame({'PassengerId':test_ids,"Survived":y_test})
# test.to_csv('/home/sirius/project/python/Kaggle/Titanic/data/res.csv',index=False)
