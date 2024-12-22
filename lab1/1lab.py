import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
nltk.download('punkt_tab')
nltk.download('punkt')

# Завантаження даних
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    return data

# Попередня обробка текстів
def preprocess_text(text):
    # Видалення спеціальних символів
    text = re.sub(r'[^\w\s]', '', text)
    # Токенізація
    tokens = word_tokenize(text.lower())
    # Видалення стоп-слів
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Стемінг
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Основний код
positive_data = load_data('plot.tok.gt9.5000')
negative_data = load_data('quote.tok.gt9.5000')

# Позначення класів
positive_labels = [1] * len(positive_data)
negative_labels = [0] * len(negative_data)

# Об'єднання даних
texts = positive_data + negative_data
labels = positive_labels + negative_labels

# Попередня обробка текстів
texts = [preprocess_text(text) for text in texts]

# Векторизація текстів
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
y = np.array(labels)

# Розподіл на тренувальну і тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Модель логістичної регресії
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Оцінка моделі
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Створення словника частотності слів для кожного класу
def build_frequency_dict(texts, labels):
    positive_texts = [texts[i] for i in range(len(labels)) if labels[i] == 1]
    negative_texts = [texts[i] for i in range(len(labels)) if labels[i] == 0]

    positive_words = ' '.join(positive_texts).split()
    negative_words = ' '.join(negative_texts).split()

    positive_freq = Counter(positive_words)
    negative_freq = Counter(negative_words)

    return positive_freq, negative_freq

# Виклик функції для побудови словників
positive_freq, negative_freq = build_frequency_dict(texts, labels)

# Виведення кількох найчастіших слів
print("Топ-10 слів для позитивного класу:")
print(positive_freq.most_common(10))

print("\nТоп-10 слів для негативного класу:")
print(negative_freq.most_common(10))

# Налаштування параметрів для перебору
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs']
}

# Модель і перебір параметрів
grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Найкращі параметри
print("\nНайкращі параметри:", grid_search.best_params_)

# Оцінка моделі з найкращими параметрами
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точність моделі з найкращими параметрами: {accuracy:.2f}")
