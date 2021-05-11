import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

dataset = pd.read_csv('data/winequality-red.csv')

dataset.columns = [i.replace(' ', '_') for i in dataset.columns]

margin_to_ten = 10 - dataset.quality.max()
dataset.quality = [i + margin_to_ten for i in dataset.quality]

X = dataset[dataset.columns[:-1].tolist()]
y = dataset['quality']

regressor = LinearRegression()

regressor.fit(X, y)

pickle.dump(regressor, open('model.pkl', 'wb'))

print('\033[92m' + 'Model is ready to go!!!' + '\033[92m')
