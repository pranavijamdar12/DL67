# Q1
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

print("-"*50)
# create the dataset

X = np.array([
  [25,500,12,1,2],
  [30,700,24,0,1],
  [45,1200,6,5,10],
  [45,1500,5,6,10],
  [50,1500,5,6,10],
  [28,600,18,1,1],
  [35,800,30,0,0],
  [48,1400,4,7,9],
  [52,1600,3,8,12],
  [27,550,20,0,1],
  [42,1300,8,4,7]
])

Y = np.array([
  0,0,1,1,0,
  0,1,1,0,1,0
])

# Clean the dataset
print("Missing Values:",np.isnan(X).sum())

# Split Data set

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)

# Apply StandardScalar
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

# create FNN Model
model = MLPClassifier(
  hidden_layer_sizes=(10,5),
  activation='relu',
  solver="lbfgs",
  max_iter=2000,
  random_state=42
)

# Train the model

model.fit(X_train,Y_train)

# Evaluate the model

y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test,y_pred)
print("Model Accuracy :",accuracy*100,"%")

new_customer = np.array([[
  46,1450,5,6,9
]])

new_customer_scaled = scalar.transform(new_customer)
prediction = model.predict(new_customer_scaled)
if prediction[0] == 0:
  print("Predication:Customer will stay")
else:
  print("Predication:Customer may leave")





