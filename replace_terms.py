import os
import json

def main():
    # Определение путей к каталогам и файлу словаря относительно скрипта
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, 'terms_map.json')
    source_dir = os.path.join(base_dir, 'wiki_first')
    target_dir = os.path.join(base_dir, 'knowledge_base')

    # Проверка наличия словаря
    if not os.path.exists(json_path):
        print(f"Ошибка: Файл словаря {json_path} не найден.")
        return

    # Загрузка словаря из JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        terms_map = json.load(f)

    # Сортируем ключи по длине по убыванию. Это нужно, чтобы длинные термины 
    # (например, с ".md") заменялись первыми, не допуская частичной замены текста
    sorted_keys = sorted(terms_map.keys(), key=len, reverse=True)

    # Создание целевого каталога, если он не существует
    os.makedirs(target_dir, exist_ok=True)

    if not os.path.exists(source_dir):
        print(f"Внимание: Исходный каталог {source_dir} не найден. Создайте его и добавьте файлы.")
        return

    # Обработка файлов
    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)
        
        if not os.path.isfile(file_path):
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Выполнение замен в содержимом файла и в имени файла
        new_filename = filename
        for key in sorted_keys:
            content = content.replace(key, terms_map[key])
            new_filename = new_filename.replace(key, terms_map[key])

        target_file_path = os.path.join(target_dir, new_filename)
        with open(target_file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"Файл {filename} успешно обработан и сохранен как {new_filename}")

if __name__ == "__main__":
    main()