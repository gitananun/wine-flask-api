import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

dataset = pd.read_csv('data/winequality-red.csv')

dataset.columns = [i.replace(' ', '_') for i in dataset.columns]

margin_to_ten = 10 - dataset.quality.max()
dataset.quality = [i + margin_to_ten for i in dataset.quality]

X = dataset[dataset.columns[:-1].tolist()]
y = dataset['quality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.33,
    random_state=101
)

regressor = LinearRegression()

regressor.fit(X_train, y_train)

pickle.dump(regressor, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))

print('\033[92m' + 'Model is ready to go!!!' + '\033[92m')
