import cv2 as cv
import numpy as np
import os
import pylab as pl
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, vq
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler


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
def resizing_img(img, new_width=500, interp=cv.INTER_AREA):
    h, w = img.shape[:2]

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
# im = cv.imread(image_paths[1])
# im = resizing_img(im)
# orb = cv.ORB_create()
# # Построение ключевых точек
# kp = orb.detect(im, None)
# kp, des = orb.compute(im, kp)
# img = draw_keypoints(im, kp)
# plt.show()

# Добавление дескрипторов обучающих изображений в список
orb = cv.ORB_create()

for image_pat in image_paths:
    im = cv.imread(image_pat)

    im = resizing_img(im)
    kp = orb.detect(im, None)
    keypoints, descriptor = orb.compute(im, kp)
    des_list.append((image_pat, descriptor))

im = cv.imread(image_paths[4])
im = resizing_img(im)
cv.imshow('test',im)

descriptors = des_list[0][1]

for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

# Выполнение кластеризации K-средних по дескрипторам
descriptors_float = descriptors.astype(float)
k = 200
voc, variance = kmeans(descriptors_float, k, 1)

# Создание гистограммы обучающего изображения
im_features = np.zeros((len(image_paths), k), "float32")

for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1], voc)
    for w in words:
        im_features[i][w] += 1

# Применение стандартизации в функции обучения
stdslr = StandardScaler().fit(im_features)
im_features = stdslr.transform(im_features)

# Создание модели классификации с помощью SVM
clf = LinearSVC(max_iter=80000)
clf.fit(im_features, np.array(y_train))

# Тестирование модели классификации
des_list_test = []

for image_pat in image_paths_test:
    image = cv.imread(image_pat)
    im = resizing_img(image)
    kp = orb.detect(image, None)
    keypoints_test, descriptor_test = orb.compute(image, kp)
    des_list_test.append((image_pat, descriptor_test))

test_features = np.zeros((len(image_paths_test), k), "float32")

for i in range(len(image_paths_test)):
    words, distance = vq(des_list_test[i][1], voc)
    for w in words:
        test_features[i][w] += 1

test_features = stdslr.transform(test_features)

true_classes = []
for i in y_test:
    if i == 'Deer':
        true_classes.append("Олень")
    elif i == 'Penguin':
        true_classes.append("Пингвин")
    else:
        true_classes.append("Пикачу")

predict_classes = []

for i in clf.predict(test_features):
    if i == 'Deer':
        predict_classes.append("Олень")
    elif i == 'Penguin':
        predict_classes.append("Пингвин")
    else:
        predict_classes.append("Пикачу")


print(Dataset)
print(get_all_elements_in_list(Dataset))
print(len(train))

cv.waitKey()
