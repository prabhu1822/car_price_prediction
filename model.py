# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df = pd.read_csv("C:\Development\Classwork\Projects\Car price prediction\cmlr.csv")

x = df.iloc[:, 1:3]
y = df.iloc[:, 0]


#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
model = LinearRegression()

#Fitting model with trainig data
model.fit(x, y) 

#y=m1x1+m2x2+c

# Saving model to disk
pickle.dump(model, open('lin_reg_model.pkl','wb'))

'''
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))
'''