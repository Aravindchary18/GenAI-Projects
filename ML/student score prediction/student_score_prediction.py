# Import NumPy for handling arrays
import numpy as np 
# Import LinearRegression model from scikit-learn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 


# Feature: study hours (input)
X=np.array([[1],[2],[3],[4],[5]])

# Target: exam scores (output)
y=np.array([50,55,65,70,80])

# Create Linear Regression model
model=LinearRegression()


# Train the model with input and output
model.fit(X,y)

# New sample: study 6 hours
new_study_hour=[[6]]


# Predict exam score for new sample
prediction=model.predict(new_study_hour)


# Print predicted score
print("prediction score:",prediction[0])