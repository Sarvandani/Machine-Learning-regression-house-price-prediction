#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:41:58 2023

@author: Sarvandani
"""
import pandas as pd
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

dataset = pd.read_csv("Housing.csv")
fig = px.scatter_3d(x=dataset['price'], y=dataset['bedrooms'], z=dataset['bathrooms'], color=dataset['price'], title="3D_Info")
fig.update_layout(scene = dict(
                    xaxis_title='price',
                    yaxis_title='bedrooms',
                    zaxis_title='bathrooms'),
                    width=1000,
                    margin=dict(r=20, b=10, l=10, t=10))
fig.show()
# we convert the price to the list
y = list(dataset['price'])
# see the names of all the columns
columns = dataset.columns
# features
X = dataset[columns[0:5]]
#lable
y = dataset[columns[0]]
# we split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
xgb.XGBRegressor().get_params()
xg_reg = xgb.XGBRegressor()
# we train the model
xg_reg.fit(X_train,y_train)
xgb_preds = xg_reg.predict(X_test)
plt.figure(figsize=(20, 8))
# plotting the graphs
plt.plot([i for i in range(len(y_test))],y_test, label="Real Price")
plt.plot([i for i in range(len(y_test))],xgb_preds, label="Predicted Price")
# showing the plot
plt.legend()
plt.show()
# We evaluate the model
print('R score is :', r2_score(y_test, xgb_preds))

