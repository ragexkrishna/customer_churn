import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import preprocess_data
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

X,y = preprocess_data("data/telecom_churn.csv")

X_train,X_test,y_train,y_test = train_test_split(
    X,y, test_size=0.2,random_state=42
)

model = GradientBoostingClassifier()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("confusion matrix")
print(cm)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

with open("model/churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved successfully!")
