from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.metrics import confusion_matrix
from catboost import CatBoostClassifier

pd_list = []
pf_list = []
bal_list = []
fir_list = []


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
    FIR = (PD - FI) / PD
    print('FIR : ', FIR)

    return PD, PF, balance, FIR


data = np.loadtxt("C:/Users/sojeong/Desktop/revisiting-models/data/zxing/zxing.csv", delimiter=",", skiprows=1, dtype=np.float32)
x = data[:, :26]
y = data[:, 26]

# 교차 검증
kf = StratifiedKFold(n_splits=10, shuffle=False)
for train_index, test_index in kf.split(x, y):
    N_train, N_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

# 정규화
    scaler = MinMaxScaler()
    N_train[:] = scaler.fit_transform(N_train[:])
    N_test[:] = scaler.fit_transform(N_test[:])

    # SMOTE(train 데이터만 진행)
    smote = SMOTE(random_state=1004)
    N_train_over, y_train_over = smote.fit_resample(N_train, y_train)

    sklearn_xgboost_model = CatBoostClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
    sklearn_xgboost_model.fit(N_train_over, y_train_over)
    y_pred = sklearn_xgboost_model.predict(N_test)

    print("catboost : ")
    PD, PF, bal, FIR = classifier_eval(y_test, y_pred)

    pd_list.append(PD)
    pf_list.append(PF)
    bal_list.append(bal)
    fir_list.append(FIR)

print('avg_PD: {}'.format((sum(pd_list) / len(pd_list))))
print('avg_PF: {}'.format((sum(pf_list) / len(pf_list))))
print('avg_balance: {}'.format((sum(bal_list) / len(bal_list))))
print('avg_FIR: {}'.format((sum(fir_list) / len(fir_list))))
