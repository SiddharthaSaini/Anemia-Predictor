# anemia_model.py

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('anemia.csv')

# Display dataset info
print("\nDataset Info:")
print(data.info())
print("\nShape:", data.shape)

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Describe dataset
print("\nStatistical Summary:")
print(data.describe())

# Result distribution before balancing
print("\nOriginal Result Distribution:")
print(data['Result'].value_counts())
data['Result'].value_counts().plot(kind='bar', color=['blue', 'green'])
plt.title('Original Distribution of Result')
plt.xlabel('Result')
plt.ylabel('Count')
plt.show()

# Gender distribution
print("\nGender Distribution:")
print(data['Gender'].value_counts())
data['Gender'].value_counts().plot(kind='bar', color=['orange', 'green'])
plt.title('Distribution of Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Hemoglobin distribution
sns.displot(data['Hemoglobin'], kde=True)
plt.title('Distribution of Hemoglobin')
plt.xlabel('Hemoglobin')
plt.ylabel('Count')
plt.show()

# Bar plot of Hemoglobin by Gender and Result
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=data, x='Gender', y='Hemoglobin', hue='Result')
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
plt.title('Hemoglobin by Gender and Result')
plt.show()

# Pairplot
sns.pairplot(data, hue='Result')
plt.show()

# Heatmap of correlations
sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', linewidths=0.2)
plt.gcf().set_size_inches(10, 8)
plt.title('Correlation Heatmap')
plt.show()

# Feature/Target split
X = data.drop("Result", axis=1)
y = data["Result"]

# Train-test split from the original imbalanced data. Stratify to maintain distribution.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20, stratify=y)
print("\nTrain/Test Split Shapes (before balancing):")
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Downsample the majority class in the TRAINING set only to prevent data leakage
from sklearn.utils import resample

# Combine training data for resampling
train_data = pd.concat([X_train, y_train], axis=1)

majorclass = train_data[train_data['Result'] == 0]
minorclass = train_data[train_data['Result'] == 1]

major_downsampled = resample(majorclass, replace=False, n_samples=len(minorclass), random_state=42)
balanced_train_data = pd.concat([major_downsampled, minorclass])

print("\nBalanced Training Set Result Distribution:")
print(balanced_train_data['Result'].value_counts())

X_train = balanced_train_data.drop("Result", axis=1)
y_train = balanced_train_data["Result"]

# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # Fit and transform the training data
X_test = scaler.transform(X_test)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)
print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)
print("\nNaive Bayes Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# SVM
from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train, y_train)
y_pred = svc_model.predict(X_test)
print("\nSVM Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
gbc_model = GradientBoostingClassifier()
gbc_model.fit(X_train, y_train)
y_pred = gbc_model.predict(X_test)
print("\nGradient Boosting Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Test Prediction Example
sample_input = [[0, 11.6, 22.3, 30.9, 74.5]]
# Scale the sample input before prediction
scaled_sample_input = scaler.transform(sample_input)
prediction = gbc_model.predict(scaled_sample_input)
print("\nSample Prediction:", prediction[0])
if prediction[0] == 0: 
    print("You don't have any Anemic Disease")
else:
    print("You have Anemic Disease")

# Save the model and scaler
with open('model.pkl', 'wb') as model_file:
    pickle.dump(gbc_model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and Scaler saved successfully!")
