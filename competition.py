import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

#Reading the data.
df = pd.read_csv('challenge_dataset.txt',names=['x','y'])
x_values = df[['x']]
y_values = df[['y']]

#Fitting the data with a linear model.
y_reg = linear_model.LinearRegression()
y_reg.fit(x_values,y_values)
y_predict = y_reg.predict(x_values)

#Plotting the data.
plt.scatter(x_values,y_values)
plt.plot(x_values,y_predict)
plt.show()

#Printing out the model score.
score = y_reg.score(x_values,y_values)
print "The model has a score of %f" %score

#calculating pointwise error
error =  y_values.as_matrix() - y_predict
abs_error = abs(y_values.as_matrix() - y_predict)

#calculating the average error of model.
ave_error_total = np.mean(abs_error)
print "The average error of the model is %f" % ave_error_total

#Visualize the pointwise error.
plt.scatter(x_values,error)
plt.show()




