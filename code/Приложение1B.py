import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("ndvi_values.csv")
plt.hist(data['NDVI'], bins=30, color='green', edgecolor='black')
plt.title("NDVI Histogram")
plt.xlabel("NDVI Value")
plt.ylabel("Frequency")
plt.savefig("ndvi_histogram.png")
plt.show()