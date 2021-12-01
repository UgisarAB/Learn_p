import os


# Функция для перечисления всех имен файлов в каталоге
def img_list(path):
    return (os.path.join(path, f) for f in os.listdir(path))


# Функция подсчёта всех элементов в списке, включая вложенные списки
def get_all_elements_in_list(lst):
    count = 0
    for element in lst:
        count += len(element)
    return count


train_path = "dataset"  # каталог  с классами
class_names = os.listdir(train_path)  # Имена классов по названию папок

D, image_paths = [[] for _ in range(len(class_names))], [[] for _ in range(len(class_names))]

for i in range(len(class_names)):  # Путь до каждой отдельной картинки в каждом классе
    dir_ = os.path.join(train_path, class_names[i])
    class_path = img_list(dir_)
    image_paths[i].extend(class_path)

# Присваивание каждой картинке своего класса
for i in range(len(D)):
    for j in range(len(image_paths[0])):
        D[i].append((image_paths[i][j], class_names[i]))

# Разбиение списка на тренировочную и тестовую выборки
dataset = D

train = dataset[:36]
test = dataset[36:]

# image_paths, y_train = zip(*train)
# image_paths_test, y_test = zip(*test)

print(image_paths)
print()
print(D)




