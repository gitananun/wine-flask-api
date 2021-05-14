import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

wine1 = pd.read_excel('./data/wine-data-2.xlsx')
wine2 = pd.read_csv('./data/winequality-red.csv')

dataset = pd.merge(wine1, wine2, how="outer")

dataset.columns = [i.replace(' ', '_') for i in dataset.columns]
dataset.drop_duplicates(inplace=True)

margin_to_ten = 10 - dataset.quality.max()
dataset.quality = [i + margin_to_ten for i in dataset.quality]

X = dataset[dataset.columns[:-1].tolist()]
y = dataset['quality']

regressor = LinearRegression()

regressor.fit(X, y)

pickle.dump(regressor, open('model.pkl', 'wb'))

print('\033[92m' + 'Model is ready to go!!!' + '\033[92m')
