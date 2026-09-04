import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score
from sklearn.tree import DecisionTreeClassifier
import joblib


# Load dataset
df=pd.read_csv("train.csv")

# Encode categorical feature
df["Sex"]=df["Sex"].map({"male":0,"female":1})

# Handle missing values
df["Age"]=df["Age"].fillna(df["Age"].mean())
df["Fare"]=df["Fare"].fillna(df["Fare"].mean())

# Define features and target
X=df[["Pclass","Sex","Age","Fare"]]
y=df["Survived"]


# Split data into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=19)


# Train Logistic Regression model
lr_model=LogisticRegression(max_iter=1000)
lr_model.fit(X_train,y_train)

y_pred=lr_model.predict(X_test)

print("\n=== Logistic Regression ===")
print(f"Accuracy: {accuracy_score(y_test,y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test,y_pred)}")
print(f"Classification Report:\n{classification_report(y_test,y_pred)}")


# Train Decision Tree model
dt_model=DecisionTreeClassifier(random_state=33)
dt_model.fit(X_train,y_train)
dt_pred=dt_model.predict(X_test)
print("\n=== Decision Tree ===")
print(f"Accuracy: {accuracy_score(y_test,dt_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test,dt_pred)}")
print(f"Classification Report:\n{classification_report(y_test,dt_pred)}")

# Select Best Model (F1 Score)
lr_f1=f1_score(y_test,y_pred)
dt_f1=f1_score(y_test,dt_pred)

best_model=lr_model if lr_f1>dt_f1 else dt_model

# Save best model
joblib.dump(best_model,"model.pkl")

print("\nBest model saved based on F1-score")
