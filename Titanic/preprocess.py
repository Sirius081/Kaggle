from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np
import pandas as pd
from general import HelperFunction as hf
# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.feature_selection import RFECV
print("load data..")
# load data
train = pd.read_csv("/home/sirius/project/python/Kaggle/Titanic/data/train.csv")
test = pd.read_csv("/home/sirius/project/python/Kaggle/Titanic/data/test.csv")
full = train.append(test, ignore_index=True)
titanic = full[:891]
del train,test

titanic.describe()
hf.plot_correlation_map(titanic)  # correlation
hf.plot_distribution(titanic, 'Age', 'Survived', row='Sex')

print("extract features")
# feature extraction(pclass,sex,age,family_size(sibsp,parch),fare,embarked,(title,cabin))
pclass = pd.get_dummies(full.Pclass, prefix='Pclass')
sex = pd.Series(np.where(full.Sex == 'male', 1, 0), name='Sex')
age=full.Age.fillna(full.Age.mean())

family=pd.DataFrame()
family['F_size']=full.SibSp+full.Parch+1
family['F_Single']=family['F_size'].map(lambda n:1 if n==1 else 0)
family['F_Small']=family['F_size'].map(lambda n:1 if 2<=n<=4 else 0)
family['F_Large']=family['F_size'].map(lambda n:1 if n>=5 else 0)
fare=full.Fare.fillna(full.Fare.mean())
embarked=pd.get_dummies(full.Embarked,'Embarked')

Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"

                    }
title=pd.DataFrame()
title['Title']=full['Name'].map(lambda n:n.split(',')[1].split('.')[0].strip())
title['Title']=title.Title.map(Title_Dictionary)
title['Title']=pd.get_dummies(title.Title,prefix='Title')
cabin=full.Cabin.fillna('U')
cabin=cabin.map(lambda c:c[0])
cabin=pd.get_dummies(cabin,prefix='Cabin')

fea=pd.concat([pclass,sex,age,family,fare,embarked,title,cabin],axis=1)

