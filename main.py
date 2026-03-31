import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")
print(df.head())
prod_per_year = df.groupby('year').totalprod.mean().reset_index()
X = prod_per_year['year']
X = X.values.reshape(-1, 1)
y= prod_per_year['totalprod']
regr = linear_model.LinearRegression()
regr.fit(X,y)
print("Slope:", regr.coef_)
print("Intercept:", regr.intercept_)
y_predict = regr.predict(X)
plt.scatter(X,y)
plt.plot(X,y_predict)
plt.xlabel('Year')
plt.ylabel('Total Honey Production')
plt.title('Linear Regression: Honey Production vs Year')
plt.show()
X_future = np.array(range(2013, 2051))
X_future = X_future.reshape(-1, 1)
future_predict = regr.predict(X_future)
plt.plot(X_future,future_predict)
plt.plot(X_future, future_predict)
plt.xlabel("Year")
plt.ylabel("Total Honey Production")
plt.title("Predicted Honey Production (2013–2050)")
plt.show()
print(future_predict[-1])
