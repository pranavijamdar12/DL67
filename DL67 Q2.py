import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# Q1: Create Dataset
X = np.array([
    [25000, 600, 200000, 10000, 0],
    [40000, 700, 300000, 800, 1],
    [60000, 750, 500000, 12000, 1],
    [20000, 550, 150000, 15000, 0],
    [80000, 800, 700000, 10000, 1],
    [35000, 650, 250000, 9000, 1],
    [18000, 500, 100000, 12000, 0],
    [90000, 850, 800000, 15000, 1],
    [30000, 580, 200000, 14000, 0],
    [70000, 780, 600000, 10000, 1]
])


Y = np.array([
    0, 1, 1, 0, 1,
    1, 0, 1, 0, 1
])


# Q2: Convert last column into integer
X[:, 4] = X[:, 4].astype(int)


# Q3: Split Dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)


print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("Y_train:", Y_train.shape)
print("Y_test:", Y_test.shape)


# Q4: Apply Scaling
scalar = StandardScaler()

X_train = scalar.fit_transform(X_train)

X_test = scalar.transform(X_test)


# Q5: Create FNN Model
model = MLPClassifier(
    hidden_layer_sizes=(10, 5),
    activation='relu',
    solver='lbfgs',
    max_iter=2000,
    random_state=42
)


# Q6: Train Model
model.fit(X_train, Y_train)


# Q7: Predict Test Data
Y_pred = model.predict(X_test)


# Q8: Calculate Accuracy
accuracy = accuracy_score(Y_test, Y_pred)

print("Model Accuracy:", accuracy * 100, "%")


# Q9: Predict New Applicant
new_applicant = np.array([
    [55000, 720, 400000, 10000, 1]
])


# Apply same scaling
new_applicant_scaled = scalar.transform(new_applicant)


# Prediction
prediction = model.predict(new_applicant_scaled)


if prediction[0] == 0:
    print("Prediction: Loan Rejected")
else:
    print("Prediction: Loan Approved")
