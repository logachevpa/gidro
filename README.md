# Алгоритм

Для каждой строчки файла производиться интерполяция на двух следующих, и считается корреляцию со следующей, далее при накоплении достаточного количества обработанных строчек вычитаем скользящее среднее, далее полученный сигнал  с его спектром визуализуализируется

# Установка

- Установить python3 либо из официального репозитория либо как часть проекта anaconda

- Установить зависимости
```
pip3 install -r requirements.txt
```

# Запуск
```
cd hydrophone
python main.py  --path ../../data/raw_2018.02.15_11.02.38.raw --bound 0 3700 768 782 --zoom 10 --window 100
```

Параметры запуска
```
--path - путь до raw файла
--output - путь до выходной картинке (по умолчанию ../figs/fig/fig_datetime.png)
--zoom - параметр интерполяция сигнала, чем больше тем точнее синусоида, но медленней, 
так как увеличивает количество точек для обработки в zoom^2 раз 
--bound - границы области (top, bottom, left, right)
--window - окно скользящего среднего, для устранения смещения синусоиды 