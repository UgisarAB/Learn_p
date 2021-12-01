import os


# Функция для перечисления всех имен файлов в каталоге
def img_list(path):
    return (os.path.join(path, f) for f in os.listdir(path))


image_paths = []
image_classes = []
D = []

train_path = "dataset"  # каталог  с классами

class_names = os.listdir(train_path)  # Имена классов по названию папок

for training_name in class_names:  # Путь до каждой отдельной картинки в каждом классе
    dir_ = os.path.join(train_path, training_name)
    class_path = img_list(dir_)
    image_paths += class_path

# Присваивание каждой картинке своего класса
for training_name in class_names:
    image_classes.extend([f'{training_name}'] * (len(image_paths) // 3))

for i in range(len(image_paths)):  # Формирование списка D из пары путь + класс
    D.append((image_paths[i], image_classes[i]))

# Разбиение списка на тренировочную и тестовую выборки
dataset = []

train = dataset[:36]
test = dataset[36:]

image_paths, y_train = zip(*train)
image_paths_test, y_test = zip(*test)

print(image_classes)
print(len(image_classes))

# print()
print(len(D))
# print(train)
# print(test)
#
# print()
# print(len(D))
# print(class_names)
# print(image_classes)
# print(len(image_paths))
