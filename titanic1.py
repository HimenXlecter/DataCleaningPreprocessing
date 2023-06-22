import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
plt.ioff()

# Veri okuma
data = pd.read_csv('titanic.csv')

# Anlamsız sütunları silme
data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Kategorik özellikleri one-hot encoding yapma
data = pd.get_dummies(data, columns=['Sex', 'Embarked'])

# Eksik verileri doldurma
imputer = SimpleImputer(strategy="mean")
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Öznitelikleri ve hedefi belirleme
features = ["Pclass", "Sex_male", "Sex_female", "SibSp", "Parch", "Embarked_C", "Embarked_Q", "Embarked_S"]
X = data_imputed[features]
y = data_imputed["Survived"]

# Veriyi eğitim ve test setlerine bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model oluşturma
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Tahminler yapma
predictions = model.predict(X_test)

# Doğruluk oranını hesaplama
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy}")

# Koralasyon matrisini oluşturma
corr_matrix = data.corr()

# Koralasyon matrisini görselleştirme
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Kategorik öznitelikler için barplot çizme
for feature in ['Sex_male', 'Sex_female', 'Embarked_C', 'Embarked_Q', 'Embarked_S']:
    plt.figure(figsize=(10,4))
    sns.barplot(data=data, x=feature, y='Survived')
    plt.show()
