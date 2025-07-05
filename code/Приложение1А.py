import rasterio
import numpy as np
import matplotlib.pyplot as plt

with rasterio.open("DJI_0016.TIF") as src:
    red = src.read(3).astype(float)
    nir = src.read(4).astype(float)

    ndvi = (nir - red) / (nir + red + 1e-6)
    plt.imshow(ndvi, cmap='RdYlGn')
    plt.colorbar(label='NDVI')
    plt.title("NDVI Map of DJI_0016.TIF")
    plt.savefig("ndvi_map.png")
    plt.show()
