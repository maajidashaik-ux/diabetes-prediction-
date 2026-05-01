import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score,precision_score,f1_score

data = pd.read_csv(r"C:\Users\maaji\Downloads\diabetes.csv")

x = data.drop(columns="Outcome")
y = data["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train),columns=x.columns)
x_test_scaled = scaler.transform(x_test)

model = LogisticRegression()
model.fit(x_train_scaled, y_train)

print(model.score(x_test_scaled,y_test))

from sklearn.model_selection import cross_val_score

scores=cross_val_score(model,x_test_scaled,y_test,cv=5)
print("Scores:",scores.mean())