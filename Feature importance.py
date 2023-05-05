from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,GradientBoostingRegressor,ExtraTreesRegressor
import numpy as np
import pandas as pd

dataset= r'C:\Users\86151\Desktop\Molecular_Descriptor.csv'

data=pd.DataFrame(pd.read_csv(dataset))
featureData=data.values[:,0:-1]
print(type(featureData),featureData.shape)
homo= data.values[:, -1]
featurename=pd.read_csv(dataset)[0:0:-1]
print(featurename)
rf = ExtraTreesRegressor(n_estimators=150)
rf.fit(featureData, homo)
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), featurename), reverse=True))
