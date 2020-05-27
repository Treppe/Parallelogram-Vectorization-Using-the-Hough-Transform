## Структура алгоритма

------

Алгоритм состоит из 2-х этапов:

1. Поиск на изображении (в нашем случае наборе точек) прямых линий используя процедуру голосования. 
2. Поиск 4-х линий образующие искомый параллелограмм и его вершины.

## Комментарии к использованию алгоритма

------

### Константы аппроксимации

Точность и скорость работы данного алгоритма зависит преимущественно от того, какие константы аппроксимации были заданы в начале. Объяснение смысла каждой из констант и рекомендованные значения указанны ниже:

* **START_PEAK_HEIGHT_T**. 
  На этапе поиска прямых линий нам необходимо решить какое минимально кол-во точек изображения должно лежать на одной прямой (минимальна "длина" прямой), что бы её можно было рассматривать, как потенциального кандидата на роль стороны параллелограмма. В данном алгоритме это решение зависит от самой длинной найденной максимальной прямой. Допустим такая прямая проходит через **L** точек, тогда обрабатываться будут те прямые, которые прошли через кол-во точек равное **l ** = **L** * **START_PEAK_HEIGHT_T**. 
  Значение данной константы должно подбираться из соображений, что если **n** будет слишком мало, то мы получим слишком много прямых и дальнейшие вычисления займут слишком много времени. В худшем случае для **n** точек мы можем получить ***n!/(n − 2)!/2!*** прямых, если **l** = 2, т.к. через каждую пару точек можно будет провести прямую. Поэтому не рекомендуется использовать слишком низкое значение.
  Рекомендуемое значение для данной константы 1-0.9. При таком выборе алгоритм в начале попробует найти параллелограмм состоящий из самых длинных обнаруженных сторон, а если не найдётся, быстро (т.к. самых длинных линий обычно очень мало) повторит процесс, выбрав меньший коэффициент перед  для максимальной длины **L**. Сделает он это, уменьшая данный коэффициент на значение **PEAK_DEC**, до тех пор пока не найдёт параллелограмм или пока **l** не окажется меньше или равно константе **BAD_HEIGHT**. Во втором случае алгоритм прекратит поиск и сделает вывод, что на изображении нет параллелограмма. 

* **PEAK_DEC**.
  Величина на которую будет уменьшатся коэффициент перед **L** в случае, когда параллелограмм не был найден. Значение данной константы зависит от выбранного значения **START_PEAK_HEIGHT_T**. Во время тестирования использовалось значение **PEAK_DEC** = **START_PEAK_HEIGHT_T** / 10, что позволяло получать положительные результаты

* **BAD_HEIGTH.**
  Прямая, которая пересекла кол-во точек меньшее или равное, чем **BAD_HEIGTH** не будет считаться за прямую. Рекомендуемое значение: 2

  > Из вышеперечисленных констант видно, что алгоритм не распознает параллелограмм, на стороне, которого лежит менее 3 точек. Это слабое место алгоритма: он не может корректно обработать параллелограммы с большими разрывами в одной из сторон.

* **RHO_RES**, **THETA_RES**.
  Данные константы отвечают за точность алгоритма. Они определяют то, как близко должна находится точка рядом к прямой, чтобы считалось, что данная точка лежит на этой прямой. 
  Меньшее значение этих констант даёт большую точность, однако слишком низкие значения могут значительно увеличить длительность работы алгоритма. Поэтому рекомендуемое значение для этих констант: 0.5 - для относительно чистых изображений с малым разброс точек вдоль сторон; 1 - для шумных изображений. 

* **LENGTH_T**.

  Данная константа используется в процессе поиска параллельных сторон параллелограмма. Обеспечивает равенство длинных этих сторон. Ускоряет дальнейший процесс поиска всех сторон параллелограмма. Рекомендуемое значение: 0.3 - для относительно чистых изображений с малым разброс точек вдоль сторон; 0.5 - для шумных изображений. 

* **MAX_DIV**
  Максимальное допустимое суммы квадратов отклонений точек данной фигуры от найденных параллелограммов. Незначительно ускоряет дальнейший процесс поиска лучшего параллелограмма. Значение подбирается на основе требований к точности. В ходе тестирования значение равное 10 позволяло получать положительные результаты.  

### Результаты тестирования и область применения алгоритма

Алгоритм эффективно обрабатывает шумные параллелограммы, не содержащие больших разрывов. Так наборы точек из предоставленных txt фалов 1, 2, 3 и dat файлов magnet_1 - magnet_6 и coil_1 дали следующие результаты:

​													 ![1](/home/egor/Google Drive/Programming/PROJECTS/ОПД/Parallelogram-Detection-Using-the-Hough-Transform/Test Results/1.png)

![2](/home/egor/Google Drive/Programming/PROJECTS/ОПД/Parallelogram-Detection-Using-the-Hough-Transform/Test Results/2.png)

![3](/home/egor/Google Drive/Programming/PROJECTS/ОПД/Parallelogram-Detection-Using-the-Hough-Transform/Test Results/3.png)

![coil_1](/home/egor/Google Drive/Programming/PROJECTS/ОПД/Parallelogram-Detection-Using-the-Hough-Transform/Test Results/coil_1.png)

![magnet_1](/home/egor/Google Drive/Programming/PROJECTS/ОПД/Parallelogram-Detection-Using-the-Hough-Transform/Test Results/magnet_1.png)

![magnet_2](/home/egor/Google Drive/Programming/PROJECTS/ОПД/Parallelogram-Detection-Using-the-Hough-Transform/Test Results/magnet_2.png)

![magnet_3](/home/egor/Google Drive/Programming/PROJECTS/ОПД/Parallelogram-Detection-Using-the-Hough-Transform/Test Results/magnet_3.png)

![magnet_4](/home/egor/Google Drive/Programming/PROJECTS/ОПД/Parallelogram-Detection-Using-the-Hough-Transform/Test Results/magnet_4.png)

![magnet_5](/home/egor/Google Drive/Programming/PROJECTS/ОПД/Parallelogram-Detection-Using-the-Hough-Transform/Test Results/magnet_5.png)

![magnet_6](/home/egor/Google Drive/Programming/PROJECTS/ОПД/Parallelogram-Detection-Using-the-Hough-Transform/Test Results/magnet_6.png)

Значение "*Diff*" в заголовке графиков соответствует сумме квадратов отклонений точек данной фигуры от найденного параллелограмма.

Параллелограммы из файлов coil_2 - coil_6  обработать не удалось, т.к. те содержат стороны со сторонами, вдоль которых лежит 2 и менее точки. 

### Возможные расширения

------

#### Разрывы в параллелограммах

Возможное решение проблемы разрывов - искусственное заполнение в местах разрывов дополнительными точками. Функции исполняющие такое заполнение предложены в папке "*Extensions*"  в файле "*anti-gap.py*". Однако такое решение не предпочтительно, т.к. может непредсказуемо повлиять на достоверность полученного результата. 

#### Проверка высоты найденных параллелограммов

В ходе поиска всех 4-х сторон параллелограмма можно дать алгоритму установку игнорировать параллелограммы "длина" сторон которых (кол-во точке, которые пересекли стороны) умноженная на синус угла смежных сторон не равна высоте параллелограмма (данное условие соблюдается в идеальных параллелограммах).  Тогда для такого алгоритма надо будет добавить ещё одну константу аппроксимации **DIST_T**, рекомендуемое значение которой равно: 0.3 - для относительно чистых изображений с малым разброс точек вдоль сторон; 0.5 - для шумных изображений. Использовать данное решение воспользовавшись функциями из файла "*distance_threshold.py*" в папке "*Extntions*".

Данное решение не было использовано в основном алгоритме, т.к. количество точек и их плотность не позволяет уверенно принять за длину количество точек, пересечённых прямой. Однако в изображениях, где эти параметры имеют большее значение (контур заданной фигуры почти сплошной) данное расширение можно значительно увеличить скорость поиска вершин.