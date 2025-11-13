import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

temperature = np.array([20, 22, 25, 27, 30, 35, 40]).reshape(-1, 1)
sales = np.array([100, 120, 150, 180, 200, 230, 250])

model = LinearRegression()
model.fit(temperature, sales)

plt.scatter(temperature, sales)
plt.plot(temperature, model.predict(temperature), color='red')
plt.xlabel("Temperature (°C)")
plt.ylabel("Ice Cream Sales")
plt.title("Temperature vs Ice Cream Sales")
plt.show()