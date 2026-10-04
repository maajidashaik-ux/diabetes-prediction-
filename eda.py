import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read dataset
data = pd.read_csv("diabetes.csv")

# Basic information
print("Shape:", data.shape)

print("\nFirst 5 rows:")
print(data.head())

print("\nSummary Statistics:")
print(data.describe())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nOutcome Counts:")
print(data["Outcome"].value_counts())


# 1. Countplot
sns.countplot(x="Outcome", data=data)

plt.title("Outcome Count")
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.show()


# 2. Histogram
data.hist(figsize=(12, 10), bins=20)

plt.suptitle("Feature Histograms")
plt.show()


# 3. Heatmap
plt.figure(figsize=(10, 8))

sns.heatmap(data.corr(), annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.show()


# 4. Pairplot
sns.pairplot(
    data[["Glucose", "BMI", "Age", "Insulin", "Outcome"]],
    hue="Outcome"
)

plt.show()
