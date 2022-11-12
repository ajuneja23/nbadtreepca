import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
import math



df=pd.read_csv('./all_seasons.csv')
#print(df)
#print(df.columns.values.tolist())
df=df.drop(['Unnamed: 0', 'player_name', 'team_abbreviation','player_height', 'player_weight', 'college', 'country', 'draft_year', 'draft_round', 'draft_number','season'],axis=1)
#print(df)
#print(df.corr())

#print(target)
#print(features)
#NORMALIZATION
df=pd.DataFrame(MinMaxScaler().fit_transform(df),columns=['age', 'gp', 'pts', 'reb', 'ast', 'net_rating', 'oreb_pct', 'dreb_pct', 'usg_pct', 'ts_pct', 'ast_pct'])
#print(df)
y=df['age']
x=df.drop(['age'],axis=1)
#print(y)
#features=df.drop(['age'],axis=1)
#PCA
pca=PCA(n_components=3)
principalComponents = pca.fit_transform(x)

principalComponentDataFrame=pd.DataFrame(principalComponents,columns=['PC 1','PC 2','PC 3'])
#print(principalComponentDataFrame)
x_train=principalComponentDataFrame.head(math.floor(-0.2*(principalComponentDataFrame.shape[0]))).values.tolist()
x_test=principalComponentDataFrame.tail(math.ceil(0.2*(principalComponentDataFrame.shape[0]))).values.tolist()
y_train=y.head(math.floor(-0.2*y.shape[0])).values.tolist()
y_test=y.tail(math.ceil(0.2*y.shape[0])).values.tolist()

regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(x_train,y_train)
score=regressor.score(x_test,y_test)
print(score)
