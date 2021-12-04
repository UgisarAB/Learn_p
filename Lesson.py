import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

# Функция для перечисления всех имен файлов в каталоге
def img_list(path):
    return (os.path.join(path, f) for f in os.listdir(path))


# Функция подсчёта всех элементов в списке, включая вложенные списки
def get_all_elements_in_list(lst):
    count = 0
    for element in lst:
        count += len(element)
    return count


# Функция для построения ключевых точек
def draw_keypoints(vis, keypoints, color=(0, 255, 255)):
    for kp in keypoints:
        x, y = kp.pt
        plt.imshow(cv.circle(vis, (int(x), int(y)), 2, color))
    plt.show()


# Функция для масштабирования изображения
def resizing_img(img, new_width=628, new_height=628, interp=cv.INTER_AREA):
    h, w = img.shape[:2]

    if new_width is None and new_height is None:
        return img
    if new_width is None:
        ratio = new_height / h
        dimension = (int(w * ratio), new_height)
    else:
        ratio = new_width / w
        dimension = (new_width, int(h * ratio))

    res_img = cv.resize(img, dimension, interpolation=interp)
    return res_img


train_path = "dataset"  # каталог  с классами
class_names = os.listdir(train_path)  # Имена классов по названию папок

Dataset, image_paths = [[] for _ in range(len(class_names))], [[] for _ in range(len(class_names))]
test, train, des_list = [], [], []

for i in range(len(class_names)):  # Путь до каждой отдельной картинки в каждом классе
    dir_ = os.path.join(train_path, class_names[i])
    class_path = img_list(dir_)
    image_paths[i].extend(class_path)

# Присваивание каждой картинке своего класса
for i in range(len(Dataset)):
    for j in range(len(image_paths[0])):
        Dataset[i].append((image_paths[i][j], class_names[i]))

# Разбиение списка на тренировочную и тестовую выборки
for i in range(len(Dataset)):
    train.extend(Dataset[i][:12])
    test.extend(Dataset[i][12:])

image_paths, y_train = zip(*train)
image_paths_test, y_test = zip(*test)

print(image_paths)
# Извлечение объектов с помощью ORB
orb = cv.ORB_create()
im = cv.imread(image_paths[20])
im = resizing_img(im)

# Построение ключевых точек
kp = orb.detect(im, None)
kp, des = orb.compute(im, kp)
img = draw_keypoints(im, kp)
plt.show()


print(Dataset)
print(get_all_elements_in_list(Dataset))
print(len(train))
print(len(test))

cv.waitKey(0)