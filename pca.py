import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.decomposition import PCA




df=pd.read_csv('./all_seasons.csv')
#print(df)
#print(df.columns.values.tolist())
df=df.drop(['Unnamed: 0', 'player_name', 'team_abbreviation','player_height', 'player_weight', 'college', 'country', 'draft_year', 'draft_round', 'draft_number','season'],axis=1)
#print(df)
#print(df.corr())

#print(target)
#print(features)

df=pd.DataFrame(MinMaxScaler().fit_transform(df),columns=['age', 'gp', 'pts', 'reb', 'ast', 'net_rating', 'oreb_pct', 'dreb_pct', 'usg_pct', 'ts_pct', 'ast_pct'])
#print(df)
y=df['age']
x=df.drop(['age'],axis=1)
#print(y)
#features=df.drop(['age'],axis=1)
pca=PCA(n_components=3)
principalComponents = pca.fit_transform(x)

principalComponentDataFrame=pd.DataFrame(principalComponents,columns=['PC 1','PC 2','PC 3'])
print(principalComponentDataFrame)