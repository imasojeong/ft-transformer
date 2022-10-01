import numpy as np
import csv

import torch
from sklearn.model_selection import train_test_split

# f = open('C:/Users/sojeong/Desktop/revisiting-models/data/EQ/EQ.csv', 'r', encoding='utf-8')
# rdr = csv.reader(f)
# for line in rdr:
#     print(line)
# f.close()

# for seed in range(15):open(f'output/EQ/mlp/tuned_reproduced/{seed}.toml', 'w').write(open('output/EQ/mlp/tuning/reproduced/best.toml').read().replace('seed = 0', f'seed = {seed}'))


# EQ = np.genfromtxt("C:/Users/sojeong/Desktop/revisiting-models/data/EQ/EQ.csv", delimiter=',', skip_header=1)
# print("getfromtxt로 불러들인 EQ")
# print(EQ)
from bin.mlp import device

EQ = np.loadtxt("C:/Users/sojeong/Desktop/revisiting-models/data/EQ/EQ.csv", delimiter=",", skiprows=1, dtype=np.float32)
# print(EQ)
# EQ = np.loadtxt("C:/Users/")
# print(EQ.shape)
EQ.astype(float)
# EQ = torch.from_numpy(EQ).float()
# EQ = EQ.astype(torch.FloatTensor).to(device)
# EQ.dtype(float)
# print(EQt.dtype)
# x = EQ[:,:61]
# y = EQ[:,61]
# print(x)
# print(y)


# x = np.genfromtxt("C:/Users/sojeong/Desktop/revisiting-models/data/EQ/x.csv", delimiter=',', skip_header=1)
# # # # print("getfromtxt로 불러들인 x")
# # # print(x)
# # # print(x.dtype)
# # #
# y = np.genfromtxt("C:/Users/sojeong/Desktop/revisiting-models/data/EQ/y.csv", delimiter=',', skip_header=1)
# # print("getfromtxt로 불러들인 y")
# print(y)
# y = y.dtype(float32)
# print(y.dtype(float32))

# y = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0,
#  0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1,
#  1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0,
#  0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
#  1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1,
#  1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1,
#  0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
#  1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#  1, 0, 1, 1])
# print(y)

# 2차원 배열
# y = np.genfromtxt("C:/Users/sojeong/Desktop/revisiting-models/data/EQ/y.csv", delimiter=',', skip_header=1, dtype="float32")
# print("getfromtxt로 불러들인 y")
# print(y)
# y = np.delete(y1, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
#                    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
#                    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60], axis=1)
# print("삭제 후 y")

# 저장한 npy 파일 확인
# xx = np.load('C:/Users/sojeong/Desktop/revisiting-models/data/EQ/N_test.npy')
# yy = np.load('C:/Users/sojeong/Desktop/revisiting-models/data/EQ/y_test.npy')
# print("xx")
# print(xx)
# print("yy")
# print(yy)

# 기존 데이터 세트 확인
# c_x = np.load('C:/Users/sojeong/Desktop/revisiting-models/data/california_housing/N_test.npy')
# print(c_x)
# c_y = np.load('C:/Users/sojeong/Desktop/revisiting-models/data/california_housing/y_test.npy')
# print(c_y)

# print("loadtxt로 불러들인 y")
# y = np.loadtxt('C:/Users/sojeong/Desktop/revisiting-models/data/EQ/y.csv')
# y = np.delete(y, 0, axis=0) # 1번째 행 제거
# # y = y.data.float()
# print(y)
# print(y.dtype)

# print("file로 불러들인 y")
# data2 = []
# file2 = open('C:/Users/sojeong/Desktop/revisiting-models/data/EQ/EQ.csv', 'r', encoding='utf-8')  # file : 파일객체
# reader = csv.reader(file2)  # csv.reader(): for loop을 돌면서 line by line read
# for line in reader:
#     data2.append(line[61:])  #[제외할 열 개수:]
# file2.close()
# y2 = np.array(data2[1:])  #[제외할 행 개수:]
# y2 = y.astype(float)
# print(y2)

# train, text, val 데이터 분리
# N_train, N_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
# N_train, N_val, y_train, y_val = train_test_split(N_train, y_train, test_size=0.2, random_state=1)  # 0.25 x 0.8 = 0.2
# #
# # # npy 확장자 파일로 저장
# np.save('C:/Users/sojeong/Desktop/revisiting-models/data/EQ/N_test.npy', N_test)
# np.save('C:/Users/sojeong/Desktop/revisiting-models/data/EQ/N_train.npy', N_train)
# np.save('C:/Users/sojeong/Desktop/revisiting-models/data/EQ/N_val.npy', N_val)
# np.save('C:/Users/sojeong/Desktop/revisiting-models/data/EQ/y_test.npy', y_test)
# np.save('C:/Users/sojeong/Desktop/revisiting-models/data/EQ/y_train.npy', y_train)
# np.save('C:/Users/sojeong/Desktop/revisiting-models/data/EQ/y_val.npy', y_val)

# python bin/ft_transformer.py draftEQ/check_environment.toml
# python bin/mlp.py draftEQ/check_environment.toml
