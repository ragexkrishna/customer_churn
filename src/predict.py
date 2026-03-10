import pickle
import numpy as np

with open("model/churn_model.pkl","rb") as f:
    model = pickle.load(f)

def predict_customer(customer_data):
    data = np.array(customer_data).reshape(1,-1)
    prediction = model.predict(data)

    if prediction[0] == 1:
        return "Customer will churn"
    else: return "Customer will stay"

customer = [120, 1, 1, 2.5, 3, 300, 100, 70, 5.5, 10]
result = predict_customer(customer)

print(result)