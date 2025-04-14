import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import precision_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'C:\Users\Миша\PycharmProjects\Titanic_lab3\processed_titanic10.csv')
print(df.info())
print(df.head())

X = df.drop(columns=['Transported']) # признаки (X)
y = df['Transported'] # целевая переменная (y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42, max_depth=5)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

precision = precision_score(y_test, y_pred)
print(f'Precision: {precision:.2f}')

# Дополнительный отчёт классификации
print("Отчет классификации:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.title('Матрица ошибок - Дерево решений (Titanic)')
plt.show()

plt.figure(figsize=(20, 10))
plot_tree(clf,
          filled=True,
          feature_names=X.columns,
          class_names=['Not Transported', 'Transported'],
          rounded=True,
          proportion=True,
          precision=2)
plt.title('Дерево решений для датасета')
plt.show()