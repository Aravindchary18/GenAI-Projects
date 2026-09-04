# import required libraries
import pickle
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix

# load dataset
df=pd.read_csv("spam.csv",encoding="latin-1")

# Select columns 
df=df[["v1","v2"]]
df.columns=["label","text"]

# Convert labels
df["label"]=df["label"].map({"ham":0,"spam":1})

# Features and target 
X=df["text"]
y=df["label"]

# train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Vectorizer
vectorizer=TfidfVectorizer()
X_train_vec=vectorizer.fit_transform(X_train)
X_test_vec=vectorizer.transform(X_test)

# Model
model=LogisticRegression()
model.fit(X_train_vec,y_train)

# Prediction 
m_pred=model.predict(X_test_vec)

# Metrics 
print("Accuracy:",accuracy_score(y_test,m_pred))
print("F1 score:",f1_score(y_test,m_pred))
print("confusion matrix:\n",confusion_matrix(y_test,m_pred))

# Save model + vectorizer
pickle.dump(vectorizer,open("vectorizer.pkl","wb"))
pickle.dump(model,open("model.pkl","wb"))

print("Model and vectorizer saved sucessfully")