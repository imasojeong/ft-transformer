import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import torch

data = np.loadtxt("C:/Users/sojeong/Desktop/revisiting-models/data/zxing/zxing.csv", delimiter=",", skiprows=1, dtype=np.float32)
x = data[:, :26]
y = data[:, 26]

# 교차 검증
# kf5 = KFold(n_splits=5, shuffle=False)
# for train_index, test_index in kf5.split(x):
#     X_train, X_test = x[train_index], x[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

# i = 1
# for train_index, test_index in kf5.split(x):
#     X_train = iris_df.iloc[train_index].loc[:, features]
#     X_test = iris_df.iloc[test_index][features]
#     y_train = iris_df.iloc[train_index].loc[:, 'target']
#     y_test = iris_df.loc[test_index]['target']
#
#     # Train the model
#     model.fit(X_train, y_train)  # Training the model
#     print(f"Accuracy for the fold no. {i} on the test set: {accuracy_score(y_test, model.predict(X_test))}")
#     i += 1

# train, test, val 데이터 분리
N_trainval, N_test, y_trainval, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
N_train, N_val, y_train, y_val = train_test_split(N_trainval, y_trainval, test_size=0.2, random_state=42)  # 0.25 x 0.8 = 0.2

# 정규화
scaler = MinMaxScaler()
N_train[:] = scaler.fit_transform(N_train[:])
N_test[:] = scaler.fit_transform(N_test[:])
N_val[:] = scaler.fit_transform(N_val[:])
#
# # N_train = np.array(N_train)
# # N_test = np.array(N_test)
# # N_val = np.array(N_val)
# # y_train = np.array(y_train)
# # y_test = np.array(y_test)
# # y_val = np.array(y_val)
#
# # N_train = torch.tensor(N_train, dtype=torch.float)
# # N_test = torch.tensor(N_test, dtype=torch.float)
# # N_val = torch.tensor(N_val, dtype=torch.float)
# # y_train = torch.tensor(y_train, dtype=torch.float)
# # y_test = torch.tensor(y_test, dtype=torch.float)
# # y_val = torch.tensor(y_val, dtype=torch.float)
# #
# # print(N_train.dtype)
# # print(N_test.dtype)
# # print(N_val.dtype)
# # print(y_train.dtype)
# # print(y_test.dtype)
# # print(y_val.dtype)
#
# # N_train = torch.from_numpy(N_train)
# # N_test = torch.from_numpy(N_test)
# # N_val = torch.from_numpy(N_val)
# # y_train = torch.from_numpy(y_train)
# # y_test = torch.from_numpy(y_test)
# # y_val = torch.from_numpy(y_val)
#
# # N_train = N_train.type(torch.FloatTensor).to(device)
# # N_test = N_test.type(torch.FloatTensor).to(device)
# # N_val = N_val.type(torch.FloatTensor).to(device)
# # y_train = y_train.type(torch.FloatTensor).to(device)
# # y_test = y_test.type(torch.FloatTensor).to(device)
# # y_val = y_val.type(torch.FloatTensor).to(device)
#
# # # SMOTE(train 데이터만 진행)
smote = SMOTE(random_state=1004)
N_train_over, y_train_over = smote.fit_resample(N_train, y_train)

print('SMOTE 적용 전 학습용 피처/레이블 데이터 세트:', N_train.shape, y_train.shape)
print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트:', N_train_over.shape, y_train_over.shape)
print('SMOTE 적용 후 레이블 값 분포: \n', pd.Series(y_train_over).value_counts())
print('테스트용 피처/레이블 데이터 세트:', N_test.shape, y_test.shape)
print('검증용 피처/레이블 데이터 세트:', N_val.shape, y_val.shape)

# # npy 확장자 파일로 저장
np.save('C:/Users/sojeong/Desktop/revisiting-models/data/zxing/N_test.npy', N_test)
np.save('C:/Users/sojeong/Desktop/revisiting-models/data/zxing/N_train.npy', N_train_over)
np.save('C:/Users/sojeong/Desktop/revisiting-models/data/zxing/N_val.npy', N_val)
np.save('C:/Users/sojeong/Desktop/revisiting-models/data/zxing/y_test.npy', y_test)
np.save('C:/Users/sojeong/Desktop/revisiting-models/data/zxing/y_train.npy', y_train_over)
np.save('C:/Users/sojeong/Desktop/revisiting-models/data/zxing/y_val.npy', y_val)

# 저장된 npy 파일 클래스 분포 확인
# trainX = np.load('C:/Users/sojeong/Desktop/revisiting-models/data/EQ/N_train.npy')
# trainY = np.load('C:/Users/sojeong/Desktop/revisiting-models/data/EQ/y_train.npy')
# testX = np.load('C:/Users/sojeong/Desktop/revisiting-models/data/EQ/N_test.npy')
# testY = np.load('C:/Users/sojeong/Desktop/revisiting-models/data/EQ/y_test.npy')
# valX = np.load('C:/Users/sojeong/Desktop/revisiting-models/data/EQ/N_val.npy')
# valY = np.load('C:/Users/sojeong/Desktop/revisiting-models/data/EQ/y_val.npy')
# print('학습용 피처/레이블 데이터 세트:', trainX.shape, trainY.shape)
# print('테스트용 피처/레이블 데이터 세트:', testX.shape, testY.shape)
# print('검증용 피처/레이블 데이터 세트:', valX.shape, valY.shape)

# 실험 환경 세팅 확인
# python bin/ft_transformer.py draftEQ/check_environment.toml

# 튜닝
# cp output/EQ/ft_transformer/tuning/0.toml output/EQ/ft_transformer/tuning/reproduced.toml
# python -c "from pathlib import Path; p = Path('output/MC2/ft_transformer/tuning/reproduced.toml'); p.write_text(p.read_text().replace('n_trials = 100', 'n_trials = 5'));"
# python bin/tune.py output/zxing/ft_transformer/tuning/reproduced.toml

# 평가
# mkdir output/EQ/ft_transformer/tuned_reproduced
# python -c "for seed in range(15):open(f'output/zxing/ft_transformer/tuned_reproduced/{seed}.toml', 'w').write(open('output/zxing/ft_transformer/tuning/reproduced/best.toml').read().replace('seed = 0', f'seed = {seed}'))
# python bin/ft_transformer.py output/zxing/ft_transformer/tuned_reproduced/0.toml 0~15

# 튜닝 없이 디폴트로 평가할 때 아래 코드만 실행
# python bin/ft_transformer.py output/JDT/ft_transformer/default/0.toml 0~15

# 앙상블
# python bin/ensemble.py ft_transformer output/EQ/ft_transformer/default
