import rasterio
import numpy as np


# ---


import numpy as np
import rasterio
import matplotlib.pyplot as plt

def calculate_ndvi(image_path):
    with rasterio.open(image_path) as src:
        red = src.read(1).astype(float)
        nir = src.read(4).astype(float)
        ndvi = (nir - red) / (nir + red + 1e-10)  # предотвратяване на деление на нула
    return ndvi
red_band = src.read(3)   # например лента 3 = червена
nir_band = src.read(4)   # лента 4 = близък IR


# ---


import numpy as np
import cv2
import matplotlib.pyplot as plt


# ---


import numpy as np
import rasterio
import matplotlib.pyplot as plt

def calculate_ndvi(image_path):
    with rasterio.open(image_path) as src:
        red = src.read(1).astype(float)
        nir = src.read(4).astype(float)
        ndvi = (nir - red) / (nir + red + 1e-10)  # предотвратяване на деление на нула
    return ndvi


# ---


def plot_ndvi(ndvi_array):
    plt.figure(figsize=(8, 6))
    plt.imshow(ndvi_array, cmap='RdYlGn')
    plt.title('NDVI карта')
    plt.colorbar(label='NDVI стойности')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.show()
plt.colorbar(label='NDVI ')
plt.title('NDVI карта')
plt.show()


# ---


# Стъпка 1: Подготовка и разделяне на данни
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# indices_array съдържа входни вегетационни индекси
# yields_array съдържа съответстващи добиви

X_train, X_test, y_train, y_test = train_test_split(indices_array, yields_array, test_size=0.2, random_state=42)

# Стъпка 2: Скалиране на входни данни
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Стъпка 3: Обучение на модел
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Стъпка 4: Прогноза и оценка
y_pred = model.predict(X_test_scaled)
print("R²:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))


# ---


import cv2
import numpy as np

# Четене на отделни канали (червен и NIR) от съответни изображения
red = cv2.imread("red_channel.jpg", 0)
nir = cv2.imread("nir_channel.jpg", 0)

# Изчисление на NDVI
ndvi = (nir.astype(float) - red.astype(float)) / (nir + red + 1e-5)

# Нормализация и запис на NDVI изображение в скала 0-255
ndvi_normalized = cv2.normalize(ndvi, None, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite("ndvi_image.jpg", ndvi_normalized.astype(np.uint8))
Използвано оборудване и изходни данни
Настоящото изследване използва комбинация от високотехнологичен хардуер и софтуерни инструменти за събиране и анализ на мултиспектрални изображения. Основният сензорен носител е дрон от тип DJI Phantom 4 Pro, оборудван с RGB камера и допълнително модифициран с NIR (близко инфрачервен) канал. Камерата позволява записване на изображения с висока резолюция, необходими за точното изчисление на вегетационни индекси като NDVI. Заснемането е извършено над реални земеделски парцели в периода юни–юли 2021 година при оптимални атмосферни условия.
Използвани са следните изображения:

- DJI_0016.JPG – RGB изображение, заснето с DJI Phantom 4 Pro.
- 2021_0625_104846_140.JPG – изображение с включен инфрачервен спектър.


# ---


import cv2

image = cv2.imread("DJI_0016.JPG")
if image is None:
    raise FileNotFoundError("Файлът DJI_0016.JPG не беше открит.")
cropped = image[500:3000, 1000:4500]  # y:y+h, x:x+w
cv2.imwrite("DJI_0016_cropped.JPG", cropped)


# ---


import cv2

img = cv2.imread("DJI_0016_cropped.JPG")
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])  # Еквалиация на яркостта (Y канал)
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
cv2.imwrite("DJI_0016_equalized.JPG", img_output)


# ---


import cv2

image = cv2.imread("DJI_0016_equalized.JPG")
blue, green, red = cv2.split(image)

cv2.imwrite("red_channel.jpg", red)
cv2.imwrite("green_channel.jpg", green)
cv2.imwrite("blue_channel.jpg", blue)


# ---


import matplotlib.pyplot as plt

plt.hist(red.ravel(), bins=256, color="red", alpha=0.7)
plt.title("Хистограма на червения канал")
plt.xlabel("Интензитет")
plt.ylabel("Честота")
plt.grid(True)
plt.savefig("red_histogram.png")
plt.show()

plt.hist(green.ravel(), bins=256, color="green", alpha=0.7)
plt.title("Хистограма на зеления канал")
plt.xlabel("Интензитет")
plt.ylabel("Честота")
plt.grid(True)
plt.savefig("green_histogram.png")
plt.show()

plt.hist(blue.ravel(), bins=256, color="blue", alpha=0.7)
plt.title("Хистограма на синия канал")
plt.xlabel("Интензитет")
plt.ylabel("Честота")
plt.grid(True)
plt.savefig("blue_histogram.png")
plt.show()
Фигурата по-долу представя честотното разпределение на стойностите в червения (Red) канал на изображението. Такава хистограма позволява оценка на динамичния диапазон и преобладаващата осветеност в сцената, което е особено важно при нормализиране и сравнение на снимки, направени при различни условия.


# ---


Изображението беше предварително обработено. NDVI стойностите се изчисляват по формулата:

    NDVI = (NIR - RED) / (NIR + RED + 1e-5)

Добавянето на 1e-5 избягва деление на нула.

Алгоритъмът за изчисление на NDVI (Normalized Difference Vegetation Index) се базира на съотношението между близкоинфрачервения (NIR) и червения (RED) спектрален канал. Стойностите на NDVI варират от -1 до 1 и служат за индикатор на наличието и здравословното състояние на растителност.

Забележка: Тъй като стойностите на NDVI варират от -1 до 1, за визуализиране като изображение е необходимо NDVI картата да бъде мащабирана в диапазона [0, 255] чрез:

    ndvi_vis = ((ndvi + 1) / 2 * 255).astype(np.uint8)

След това към ndvi_vis може да се приложи цветова палитра чрез cv2.applyColorMap.
Следват основните стъпки, приложени в анализа:
- Зареждане на изображението с помощта на библиотеката OpenCV.
- Извличане на съответните спектрални канали: RED и NIR. В случая с изображението 2021_0625_104842_138.JPG, NIR каналът е разположен в третия канал (index 2), докато RED е в първия канал (index 0).
- Преобразуване на каналите до тип float, за да се избегнат целочислени деления.
- Прилагане на формулата: NDVI = (NIR - RED) / (NIR + RED + 1e-5)
- Мащабиране на резултата от [−1, 1] до диапазона [0, 255], за да се запише като изображение.
- Извеждане на NDVI картата чрез функцията cv2.imwrite().
- Ръчно маркиране на две зони („А“ и „Б“) върху NDVI картата, според визуални различия в растителността.
- Извличане на стойности на NDVI за избраните пиксели със стъпка от 50 пиксела, за намаляване на изчислителната тежест.
- Изчисляване на средна стойност, стандартно отклонение и хистограма за всяка зона поотделно.
- Потвърждаване на резултатите чрез сравнителен анализ с данни от MAPIR камера и анализ чрез специализиран софтуер.
Обобщени резултати по зони


# ---


import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("sample_pixel_data.csv")
zone_a = data[data["zone"] == "a"]["ndvi"]
zone_b = data[data["zone"] == "b"]["ndvi"]

plt.hist(zone_a, bins=20, alpha=0.6, label="Зона А")
plt.hist(zone_b, bins=20, alpha=0.6, label="Зона Б")
plt.xlabel("NDVI стойности")
plt.ylabel("Брой пиксели")
plt.legend()
plt.title("Сравнение на NDVI по зони")
plt.savefig("ndvi_hist_compare.png")

Препоръка: За да се избегне отрязване на елементи при запазване, е добре преди това да се добави: plt.tight_layout()
Геореференцираща команда (по избор)
gdal_translate -of GTiff -a_srs EPSG:4326 -a_ullr xmin ymax xmax ymin input.jpg output.tif

Пояснение: xmin, xmax, ymin, ymax следва да бъдат координати от GPS/ортомозаика, съответстващи на границите на изображението. Те трябва да се съобразят с реалната географска обвързаност на кадъра.
Тези резултати съответстват на визуалните наблюдения в NDVI картата и се потвърждават чрез ръчното зониране, използвано при анализа. Методиката демонстрира потенциала на локализирания NDVI анализ за предварителна агрономическа диагностика в прецизното земеделие (Jensen, 2007; Mulla, 2013).
Актуализация и технически усъвършенствания на скрипта
В настоящата версия на скрипта за NDVI анализ са въведени следните технически подобрения:
• Проверка за успешно зареждане на изображението;
• Подаване на входен файл чрез аргумент на командния ред;
• Автоматично зониране, базирано на позицията на пикселите;
• Подобрена стабилност при деление на нула;
• Цветова NDVI визуализация чрез applyColorMap;
• Генериране на NDVI карта и хистограма с подобрен стил.
Обновеният примерен скрипт е както следва:
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

input_path = sys.argv[1] if len(sys.argv) > 1 else "input_image.jpg"
if not os.path.exists(input_path):
    raise FileNotFoundError("Файлът не е намерен.")

img = cv2.imread(input_path)
if img is None:
    raise ValueError("Грешка при зареждане на изображението.")

red = img[:, :, 2].astype(float)
nir = img[:, :, 0].astype(float)
ndvi = (nir - red) / (nir + red + 1e-5)
ndvi_vis = ((ndvi + 1) / 2 * 255).astype(np.uint8)
ndvi_colored = cv2.applyColorMap(ndvi_vis, cv2.COLORMAP_JET)
cv2.imwrite("ndvi_map_colored.png", ndvi_colored)

rows, cols = ndvi.shape
ndvi_data = []
for i in range(0, rows, 50):
    for j in range(0, cols, 50):
        val = ndvi[i, j]
        zone = "a" if j < cols / 2 else "b"
        ndvi_data.append({"Row": i, "Col": j, "NDVI": val, "Zone": zone})

df = pd.DataFrame(ndvi_data)
df.to_csv("sample_pixel_data.csv", index=False)

data = pd.read_csv("sample_pixel_data.csv")
zone_a = data[data["Zone"] == "a"]["NDVI"]
zone_b = data[data["Zone"] == "b"]["NDVI"]

plt.figure(figsize=(8, 6))
plt.hist(zone_a, bins=20, alpha=0.6, label="Зона А", color="green")
plt.hist(zone_b, bins=20, alpha=0.6, label="Зона Б", color="orange")
plt.xlabel("NDVI стойности")
plt.ylabel("Брой пиксели")
plt.title("Сравнение на NDVI по зони")
plt.grid(True)
plt.xlim([-1, 1])
plt.legend()
plt.tight_layout()
plt.savefig("ndvi_hist_compare.png")
plt.show()
Последните технически подобрения в скрипта гарантират стабилност, възпроизводимост и автоматизация на NDVI анализа, включително възможност за задаване на входен файл, автоматично симулирано зониране и визуално подобрение на изходните карти и хистограми. Това позволява надеждно приложение в теренни условия и осигурява ефективна база за последващ агрономически анализ.


# ---


import cv2
import numpy as np

image = cv2.imread("2021_0625_104846_140.JPG")
nir = image[:, :, 0].astype(float)
red = image[:, :, 2].astype(float)

ndvi = (nir - red) / (nir + red + 1e-6)

block_size = 50
rows, cols = ndvi.shape
averages = []

for i in range(0, rows, block_size):
    for j in range(0, cols, block_size):
        block = ndvi[i:i+block_size, j:j+block_size]
        avg = np.nanmean(block)
        averages.append(avg)
Забележка: При използване на JPEG изображение от дрон, често каналът за близкия инфрачервен спектър (NIR) се намира в първи канал (индекс 0), а червеният канал (Red) – в трети (индекс 2), но това зависи от конфигурацията на камерата.
Изборът на размер 50×50 пиксела се базира на необходимостта от постигане на баланс между пространствената детайлност и ефективността на анализа. По-големи блокове намаляват влиянието на локален шум, но също така и пространствената резолюция (Mulla, 2013).
Таблица 1. Статистически показатели по зони
Изборът на блокове с размер 50×50 пиксела е направен с цел постигане на баланс между пространствена детайлност и ефективност на изчисленията. По-големи блокове намаляват локалните шумове, но също така и пространствената резолюция на NDVI картата (Mulla, 2013).
Забележка: В RGB изображения от дроновете, използвани в настоящото изследване, NIR каналът се намира на позиция 0, а Red – на позиция 2. Това се отнася до конкретния сензорен модул и може да варира при различни устройства.


# ---


Обяснение на последните подобрения в скрипта:
1. Добавена е проверка за зареждане на изображението, с цел предотвратяване на грешка при липсващ файл.
2. NDVI формулата е допълнена с защита от делене на нула чрез използване на условна конструкция np.where(...).
3. Към анализа са добавени цветни NDVI карти с цветова палитра RdYlGn, за по-добра визуална интерпретация.
4. Генерирана е допълнителна boxplot визуализация по зони, за по-информативно статистическо представяне.
5. Добавени са обяснителни коментари към всяка основна стъпка в кода.
6. Скриптът е експортиран като завършен модул и включен в Приложение 2.4.


# ---


Примерен код
# Зареждане на изображението
image = cv2.imread("2021_0625_104846_140.JPG")
nir = image[:, :, 0].astype(float)  # NIR канал
red = image[:, :, 2].astype(float)  # Red канал

# NDVI изчисление
ndvi = (nir - red) / (nir + red + 1e-6)

# Усредняване по блокове 50x50
block_size = 50
rows, cols = ndvi.shape
block_means = []
block_coords = []

for i in range(0, rows, block_size):
    for j in range(0, cols, block_size):
        block = ndvi[i:i+block_size, j:j+block_size]
        avg = np.nanmean(block)
        block_means.append(avg)
        block_coords.append((i, j))

# Създаване на таблица и добавяне на зони
df = pd.DataFrame(block_coords, columns=["Row", "Col"])
df["NDVI"] = block_means
df["Zone"] = df["Row"].apply(lambda x: "A" if x < 128 else "B")
df.to_csv("ndvi_block_50x50.csv", index=False)

# Хистограма по зони
plt.figure(figsize=(8, 4))
for zone in ["A", "B"]:
    subset = df[df["Zone"] == zone]["NDVI"]
    plt.hist(subset, bins=20, alpha=0.6, label=f"Zone {zone}")
plt.title("NDVI Histogram by Zone")
plt.xlabel("NDVI")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("ndvi_histogram_current.png")
plt.close()

# NDVI карта
ndvi_image = np.zeros_like(ndvi)
for (i, j), value in zip(block_coords, block_means):
    ndvi_image[i:i+block_size, j:j+block_size] = value
cv2.imwrite("ndvi_map_current.png", (ndvi_image * 255).astype(np.uint8))
.
Приложения:
Подробните стойности по блокове и статистическа обработка на зони A и B са представени в Приложение 2.4. Кодът е достъпен и в GitHub: https://github.com/teodoragospodinova/mul1tispectral-ndvi-analysis
Сравнителен анализ между резултатите от Python и Mapir
В рамките на това изследване беше осъществено сътрудничество с доц. д-р Аспарух Атанасов – експерт в областта на дистанционното наблюдение, геопространствения анализ и интерпретацията на вегетационни индекси. С неговото любезно съдействие беше проведен независим NDVI анализ чрез професионалната експертна система Mapir, върху същите изображения, използвани в предходните етапи на изследването.
Анализът, извършен от доц. д-р Аспарух Атанасов върху същото изображение, използва визуално обособяване на зони A и B с цел сравнение на вегетационното състояние на различни участъци от полето. Макар и да не използва фиксирана мрежа, неговият експертен подход също се базира на NDVI стойности. Резултатите от настоящия автоматизиран метод потвърждават наблюденията му, като демонстрират аналогична разлика между двете зони. Това доказва, че NDVI може успешно да бъде използван както за визуално-интерпретативен, така и за алгоритмичен анализ на растително състояние.
Сравнение на NDVI карти
На Фигура 2.5-1 са представени двете NDVI карти: а) генерирана чрез Python скрипта, и б) получена от Mapir. Визуалната съпоставка показва сходство в зоните с висока вегетация, като се отбелязват различия в динамичния обхват и контраста, което вероятно е резултат от различните алгоритми за визуализация и интерполация.
Сравнение на хистограми
Фигура 2.5-2 представя хистограмите на NDVI стойностите, получени съответно от Python и от Mapir. Макар и с различна плътност и детайлност на интервалите, двете хистограми демонстрират сходно разпределение, като зоната с най-висока честота е в диапазона между 0.3 и 0.6. Това потвърждава надеждността на скриптовия анализ.
Регресионен и статистически анализ
На базата на съпоставени NDVI стойности от двете методологии е изграден линеен регресионен модел. Резултатите показват силна положителна корелация между стойностите:
Коефициент на корелация (R): 0.835
F-статистика: 583.418
Стойност на P: < 0.0001
Анализът на ANOVA таблицата и регресионните коефициенти потвърждава, че връзката между методите е статистически значима, а отклоненията се дължат основно на различната пространствена резолюция и етапи на предварителна обработка.
Изводи
Сравнителният анализ между резултатите, получени чрез Python и Mapir, показва висока степен на съгласуваност. Това потвърждава приложимостта на автоматизирания NDVI анализ чрез скриптови средства в контекста на прецизното земеделие. Експертната система Mapir предоставя допълнителна визуална надеждност, докато Python скриптът осигурява гъвкавост и възпроизводимост


# ---


Geeks for Geeks (n.d.) *Image processing in Python*, Available at: https://www.geeksforgeeks.org/image-processing-in-python/ (Accessed: 28 June 2025).
Jensen, J.R. (2007) *Remote sensing of the environment: An earth resource perspective*. 2nd ed. Upper Saddle River, NJ: Pearson Prentice Hall.
Mulla, D.J. (2013) ‘Twenty five years of remote sensing in precision agriculture: Key advances and remaining knowledge gaps’, *Biosystems Engineering*, 114(4), pp. 358–371. doi:10.1016/j.biosystemseng.2012.08.009.
Earth Lab (n.d.) *What is NDVI?*, University of Colorado Boulder, Available at: https://www.earthdatascience.org/courses/earth-analytics/remote-sensing/vegetation-indices/ndvi/ (Accessed: 28 June 2025).
Jensen, J. R. (2007). *Remote Sensing of the Environment: An Earth Resource Perspective*. 2nd ed. Pearson Education.
Mulla, D. J. (2013). Twenty five years of remote sensing in precision agriculture: Key advances and remaining knowledge gaps. *Biosystems Engineering*, 114(4), 358–371. GeeksforGeeks. (2023). Image Processing in Python. Available at: https://www.geeksforgeeks.org/image-processing-in-python/
Earth Lab. (2020). Vegetation Indices in Remote Sensing. Available at:
Analytics Vidhya. (2021). Image Processing using NumPy with Practical Implementation and Code. Available at:
Neptune.ai. (2023). Image Processing with Python. Available at: https://neptune.ai/blog/image-processing-python


# ---


[2] Auravant (2023). Platform for Smart Agriculture. Available at: https://www.auravant.com [Accessed 30 Jun 2025].
[3] Neptune.ai (2024). How to Track Machine Learning Experiments. Available at: https://neptune.ai/blog/image-processing-python [Accessed 30 Jun 2025].
[4] Jensen, J.R. (2007). Remote Sensing of the Environment: An Earth Resource Perspective. 2nd ed. Pearson Prentice Hall.
[5] Mulla, D.J. (2013). Twenty five years of remote sensing in precision agriculture: Key advances and remaining knowledge gaps. Biosystems Engineering, 114(4), pp.358–371.
[6] Sishodia, R.P., Ray, R.L. and Singh, S.K. (2020). Applications of remote sensing in precision agriculture: A review. Remote Sensing, 12(19), p.3136.
[1] Earth Lab. (2022). Vegetation Indices. University of Colorado Boulder. https://www.earthdatascience.org/ (Accessed: 30 June 2025).
[2] Auravant. (2023). Platform for Smart Agriculture. https://www.auravant.com (Accessed: 30 June 2025).
[3] Neptune.ai. (2024). Image Processing with Python. https://neptune.ai/blog/image-processing-python (Accessed: 30 June 2025).
[4] Jensen, J. R. (2007). Remote Sensing of the Environment: An Earth Resource Perspective. 2nd ed. Pearson Prentice Hall.
[5] Mulla, D. J. (2013). Twenty five years of remote sensing in precision agriculture: Key advances and remaining knowledge gaps. Biosystems Engineering, 114(4), 358–371.
[6] Sishodia, R. P., Ray, R. L., & Singh, S. K. (2020). Applications of remote sensing in precision agriculture: A review. Remote Sensing, 12(19), 3136.


# ---


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


# ---


import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("ndvi_values.csv")
plt.hist(data['NDVI'], bins=30, color='green', edgecolor='black')
plt.title("NDVI Histogram")
plt.xlabel("NDVI Value")
plt.ylabel("Frequency")
plt.savefig("ndvi_histogram.png")
plt.show()


# ---


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


# ---


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Път до изображението
input_path = sys.argv[1] if len(sys.argv) > 1 else "input_image.jpg"
img = cv2.imread(input_path)

if img is None:
    raise FileNotFoundError("Грешка: Файлът не е намерен.")

# Извличане на канали: RED (2), NIR (0)
red = img[:, :, 2].astype(float)
nir = img[:, :, 0].astype(float)

# NDVI изчисление
ndvi = (nir - red) / (nir + red + 1e-5)

# Мащабиране до [0, 255] и оцветяване
ndvi_vis = ((ndvi + 1) / 2 * 255).astype(np.uint8)
ndvi_colored = cv2.applyColorMap(ndvi_vis, cv2.COLORMAP_JET)
cv2.imwrite("ndvi_map_colored.png", ndvi_colored)

# Извличане на стойности със стъпка
rows, cols = ndvi.shape
ndvi_data = []
for i in range(0, rows, 50):
    for j in range(0, cols, 50):
        val = ndvi[i, j]
        zone = "a" if j < cols / 2 else "b"
        ndvi_data.append({"Row": i, "Col": j, "NDVI": val, "Zone": zone})

df = pd.DataFrame(ndvi_data)
df.to_csv("sample_pixel_data.csv", index=False)

# Хистограма
data = pd.read_csv("sample_pixel_data.csv")
zone_a = data[data["Zone"] == "a"]["NDVI"]
zone_b = data[data["Zone"] == "b"]["NDVI"]

plt.figure(figsize=(8, 6))
plt.hist(zone_a, bins=20, alpha=0.6, label="Зона А", color="green")
plt.hist(zone_b, bins=20, alpha=0.6, label="Зона Б", color="orange")
plt.xlabel("NDVI стойности")
plt.ylabel("Брой пиксели")
plt.title("Сравнение на NDVI по зони")
plt.grid(True)
plt.xlim([-1, 1])
plt.legend()
plt.tight_layout()
plt.savefig("ndvi_hist_compare.png")
plt.show()


# ---


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ---


image = cv2.imread(image_path)


# ---


if image is None:
raise FileNotFoundError(f"Изображението не е намерено: {image_path}")


# ---


denominator = np.where((nir + red) == 0, 1e-6, nir + red)
ndvi = (nir - red) / denominator


# ---


for i in range(0, rows, block_size):
for j in range(0, cols, block_size):
block = ndvi[i:i+block_size, j:j+block_size]
avg = np.nanmean(block)
block_means.append(avg)
block_coords.append((i, j))


# ---


df["Zone"] = df["Row"].apply(lambda x: "A" if x < 128 else "B")


# ---


ndvi_image = np.zeros_like(ndvi)
for (i, j), value in zip(block_coords, block_means):
ndvi_image[i:i+block_size, j:j+block_size] = value


# ---


cv2.imwrite("ndvi_map.png", (ndvi_image * 255).astype(np.uint8))


# ---


normalized = (ndvi_image - np.min(ndvi_image)) / (np.max(ndvi_image) - np.min(ndvi_image))
colored_ndvi = cm.get_cmap("RdYlGn")(normalized)
cv2.imwrite("ndvi_colored.png", (colored_ndvi[:, :, :3] * 255).astype(np.uint8))


# ---


for zone in ["A", "B"]:
subset = df[df["Zone"] == zone]["NDVI"]
plt.hist(subset, bins=20, alpha=0.6, label=f"Zone {zone}")
plt.title("NDVI Histogram by Zone")
plt.xlabel("NDVI")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("ndvi_histogram.png")
plt.close()