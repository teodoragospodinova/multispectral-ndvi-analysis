import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("ndvi_yield_data.csv")
X = data[['NDVI']]
y = data['Yield']

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

plt.scatter(X, y, label="Actual")
plt.plot(X, predictions, color='red', label="Predicted")
plt.xlabel("NDVI")
plt.ylabel("Yield")
plt.title("Linear Regression: NDVI vs Yield")
plt.legend()
plt.savefig("ndvi_yield_regression.png")
plt.show()