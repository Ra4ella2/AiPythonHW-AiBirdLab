import os
import csv
import json
import sqlite3
from datetime import datetime

from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch


MODEL_NAME = "dennisjooo/Birds-Classifier-EfficientNetB2"
INPUT_FOLDER = "input"
DATABASE = "birds.db"
CSV_FILE = "bird_analysis.csv"
INFO_FILE = "bird_info.json"


#ЗАГРУЗКА МОДЕЛИ

print("Завантаження AI-моделі...")

try:
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()

    print("Модель завантажено.")

except Exception as error:
    print("Помилка завантаження моделі:")
    print(error)
    exit()


#SQLITE

db = sqlite3.connect(DATABASE)
cur = db.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS birds(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_name TEXT,
    predicted_species TEXT,
    confidence REAL,
    analysis_date TEXT,
    description TEXT,
    additional_info TEXT
)
""")

db.commit()

#ИНФОРМАЦИЯ О ПТИЦАХ

def load_bird_info():
    if not os.path.exists(INFO_FILE):
        return {}

    with open(INFO_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


bird_info = load_bird_info()


#ПРОВЕРКА ФОТО

def valid_image(path):
    if not os.path.exists(path):
        print("Файл не знайдено.")
        return False

    extension = os.path.splitext(path)[1].lower()

    if extension not in [".jpg", ".jpeg", ".png"]:
        print("Підтримуються тільки JPG, JPEG та PNG.")
        return False

    try:
        Image.open(path).verify()
        return True

    except:
        print("Не вдалося відкрити зображення.")
        return False

#AI-АНАЛИЗ

def classify_image(path):
    image = Image.open(path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    top_values, top_indexes = torch.topk(
        probabilities,
        3
    )

    results = []

    for score, index in zip(top_values, top_indexes):
        species = model.config.id2label[index.item()]

        results.append({
            "species": species,
            "confidence": score.item()
        })

    return results

#ИНФОРМАЦИОННЫЙ МОДУЛЬ

def get_info(species):
    data = bird_info.get(species)

    if not data:
        return (
            "Додаткова інформація відсутня.",
            "Немає даних у локальній базі."
        )

    description = data.get(
        "description",
        "Опис відсутній."
    )

    info = (
        f"Українська назва: {data.get('ukrainian_name', 'немає даних')}\n"
        f"Наукова назва: {data.get('scientific_name', 'немає даних')}\n"
        f"Родина: {data.get('family', 'немає даних')}\n"
        f"Середовище: {data.get('habitat', 'немає даних')}\n"
        f"Поширення: {data.get('distribution', 'немає даних')}\n"
        f"Живлення: {data.get('diet', 'немає даних')}\n"
        f"Розмір: {data.get('size', 'немає даних')}\n"
        f"Статус: {data.get('status', 'немає даних')}"
    )

    return description, info

#АНАЛИЗ ФОТО

def analyze_photo():
    image_name = input(
        "Введіть назву фотографії з папки input: "
    ).strip()

    path = os.path.join(
        INPUT_FOLDER,
        image_name
    )

    if not valid_image(path):
        return

    print("\nАналіз зображення...")

    try:
        results = classify_image(path)

    except Exception as error:
        print("Помилка аналізу:")
        print(error)
        return

    print("\n===== TOP 3 RESULTS =====")

    for i, item in enumerate(results, 1):
        print(
            f"{i}. {item['species']} "
            f"- {item['confidence'] * 100:.2f}%"
        )

    best = results[0]

    species = best["species"]
    confidence = best["confidence"]

    description, additional_info = get_info(species)

    print("\n===== BIRD INFO =====")
    print("Ймовірний вид:", species)
    print(
        "Впевненість:",
        f"{confidence * 100:.2f}%"
    )

    print("\nОпис:")
    print(description)

    print("\nДодаткова інформація:")
    print(additional_info)

    cur.execute("""
        INSERT INTO birds(
            image_name,
            predicted_species,
            confidence,
            analysis_date,
            description,
            additional_info
        )

        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        image_name,
        species,
        confidence,
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        description,
        additional_info
    ))

    db.commit()

    print("\nРезультат збережено.")
    print("ID:", cur.lastrowid)

#ИСТОРИЯ

def history():
    rows = cur.execute("""
        SELECT
            id,
            image_name,
            predicted_species,
            confidence,
            analysis_date

        FROM birds

        ORDER BY id DESC
    """).fetchall()

    print("\n===== HISTORY =====")

    if not rows:
        print("Історія порожня.")
        return

    for row in rows:
        print(
            f"ID: {row[0]} | "
            f"{row[1]} | "
            f"{row[2]} | "
            f"{row[3] * 100:.2f}% | "
            f"{row[4]}"
        )

#ПОИСК

def search_bird():
    name = input(
        "Введіть назву птаха: "
    ).strip()

    rows = cur.execute("""
        SELECT
            id,
            image_name,
            predicted_species,
            confidence,
            analysis_date

        FROM birds

        WHERE predicted_species LIKE ?
    """, (
        f"%{name}%",
    )).fetchall()

    if not rows:
        print("Птаха не знайдено.")
        return

    for row in rows:
        print(
            f"ID: {row[0]} | "
            f"Фото: {row[1]} | "
            f"Вид: {row[2]} | "
            f"Впевненість: {row[3] * 100:.2f}%"
        )

#УДАЛЕНИЕ

def delete_record():
    try:
        record_id = int(
            input("Введіть ID запису: ")
        )

    except ValueError:
        print("Некоректний ID.")
        return

    cur.execute("""
        DELETE FROM birds
        WHERE id=?
    """, (
        record_id,
    ))

    db.commit()

    if cur.rowcount:
        print("Запис видалено.")
    else:
        print("Запис не знайдено.")

#СТАТИСТИКА

def statistics():
    total = cur.execute("""
        SELECT COUNT(*)
        FROM birds
    """).fetchone()[0]

    if total == 0:
        print("Немає даних.")
        return

    most_common = cur.execute("""
        SELECT
            predicted_species,
            COUNT(*) AS count

        FROM birds

        GROUP BY predicted_species

        ORDER BY count DESC

        LIMIT 1
    """).fetchone()

    highest = cur.execute("""
        SELECT
            predicted_species,
            confidence

        FROM birds

        ORDER BY confidence DESC

        LIMIT 1
    """).fetchone()

    lowest = cur.execute("""
        SELECT
            predicted_species,
            confidence

        FROM birds

        ORDER BY confidence ASC

        LIMIT 1
    """).fetchone()

    print("\n===== STATISTICS =====")

    print(
        "Проаналізовано фотографій:",
        total
    )

    print(
        "Найчастіший вид:",
        most_common[0],
        "-",
        most_common[1],
        "разів"
    )

    print(
        "Найвища впевненість:",
        highest[0],
        f"{highest[1] * 100:.2f}%"
    )

    print(
        "Найнижча впевненість:",
        lowest[0],
        f"{lowest[1] * 100:.2f}%"
    )

#CSV

def export_csv():
    rows = cur.execute("""
        SELECT
            image_name,
            predicted_species,
            confidence,
            analysis_date,
            description

        FROM birds
    """).fetchall()

    with open(
        CSV_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "image_name",
            "predicted_species",
            "confidence",
            "analysis_date",
            "description"
        ])

        writer.writerows(rows)

    print(
        "Результати збережено у",
        CSV_FILE
    )

#MENU

while True:
    print("""
================================
          AI BIRD LAB
================================

1. Проаналізувати фотографію
2. Історія аналізів
3. Пошук птаха
4. Видалити запис
5. Статистика
6. Експорт у CSV
0. Вихід
""")

    choice = input("Оберіть пункт: ")

    if choice == "1":
        analyze_photo()

    elif choice == "2":
        history()

    elif choice == "3":
        search_bird()

    elif choice == "4":
        delete_record()

    elif choice == "5":
        statistics()

    elif choice == "6":
        export_csv()

    elif choice == "0":
        print("Програму завершено.")
        break

    else:
        print("Невірний пункт.")


db.close()