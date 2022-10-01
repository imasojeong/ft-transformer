import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.metrics import confusion_matrix, recall_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

sklearn_xgboost_model = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)

N_train = np.load("C:/Users/sojeong/Desktop/revisiting-models/data/zxing/N_train.npy")
y_train = np.load("C:/Users/sojeong/Desktop/revisiting-models/data/zxing/y_train.npy")
N_test = np.load("C:/Users/sojeong/Desktop/revisiting-models/data/zxing/N_test.npy")
y_test = np.load("C:/Users/sojeong/Desktop/revisiting-models/data/zxing/y_test.npy")

sklearn_xgboost_model.fit(N_train, y_train)

y_pred = sklearn_xgboost_model.predict(N_test)


def classifier_eval(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print('혼동행렬 : ', cm)
    PD = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    print('PD : ', PD)
    PF = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    print('PF : ', PF)
    balance = 1 - (((0 - PF) * (0 - PF) + (1 - PD) * (1 - PD)) / 2)
    print('balance : ', balance)
    FI = (cm[1, 1] + cm[0, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
    print('FI : ', FI)
    FIR = (PD - FI) / PD
    print('FIR : ', FIR)


print("xgboost : ")
classifier_eval(y_test, y_pred)

sklearn_catboost_model = CatBoostClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)

sklearn_catboost_model.fit(N_train, y_train)

y_pred = sklearn_catboost_model.predict(N_test)

print("catboost : ")
classifier_eval(y_test, y_pred)
