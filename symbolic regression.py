import numpy
import numpy as np
from gplearn.genetic import SymbolicTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from xgboost import XGBRegressor,XGBRFRegressor

dataset= r'C:\Users\86151\Desktop\CaO\Train2_12.xlsx'
data=pd.DataFrame(pd.read_excel(dataset))
featureData = data.values[:,0:12]
# band gap，即目标值
band_gap = data.values[:,-1]

# 训练Ridge模型
est = RandomForestRegressor()
est.fit(featureData, band_gap)
print(est.score(featureData, band_gap))

# lrTool = SymbolicTransformer(population_size=1000,
#                            generations=20, stopping_criteria=0.01,
#                            p_crossover=0.7, p_subtree_mutation=0.1,
#                            p_hoist_mutation=0.05, p_point_mutation=0.1,
#                            max_samples=0.9, verbose=1,tournament_size=30,
#                            parsimony_coefficient=0.0005, random_state=0,n_components=10,n_jobs=3,hall_of_fame=100)
# V=lrTool.fit(featureData,band_gap)
#
# print(V)

dataset= r'C:\Users\86151\Desktop\CaO\除法符号回归.xlsx'
data1=pd.DataFrame(pd.read_excel(dataset))

dataset= r'C:\Users\86151\Desktop\RuddlesdenPopper-\Adsorption\除法符号回归特征.xlsx'
data2=pd.DataFrame(pd.read_excel(dataset))
i = 0
j = 0
k = 0
l = 0
# for i in range(12):
#     x = data.values[:,i]
#     for j in range(12):
#         y = data.values[:, j]
#         for k in range(12):
#             m = data.values[:, k]
#             for l in range(4):
#                 n = data1.values[:, l]
#                 x = numpy.array(x)
#                 y = numpy.array(y)
#                 m = numpy.array(m)
#                 n = numpy.array(n)
#                 news = x*y+m/n
#                 lrTool = RandomForestRegressor()
#                 lrTool.fit(featureData, band_gap)
#                 Pearson = pearsonr(band_gap, news)
#                 print(i, j, k, l)
#                 if abs(Pearson[0])>0.8:
#                     print(Pearson[0])
#                 l += 1
#             k += 1
#         j += 1
#     i += 1
#     while i >= 12:
#         break



for i in range(12):
    x = data.values[:,i]
    for j in range(12):
        y = data.values[:, j]
        for k in range(12):
            m = data.values[:, k]
            x = numpy.array(x)
            y = numpy.array(y)
            m = numpy.array(m)
            news = x+y+m
            lrTool = RandomForestRegressor()
            lrTool.fit(featureData, band_gap)
            Pearson = pearsonr(band_gap, news)
            print(i, j, k)
            if abs(Pearson[0])>0.78:
                print(Pearson[0])
            k += 1
        j += 1
    i += 1
    while i >= 17:
         break

# for i in range(17):
#     x = data.values[:,i]
#     for j in range(8):
#         y = data1.values[:, j]
#         x = numpy.array(x)
#         y = numpy.array(y)
#         news = x/y
#         lrTool = RandomForestRegressor()
#         lrTool.fit(featureData, band_gap)
#         Pearson = pearsonr(band_gap, news)
#         print(i, j)
#         if abs(Pearson[0])>0.52:
#             print(Pearson[0])
#         j += 1
#     i += 1
#     while i >= 17:
#          break