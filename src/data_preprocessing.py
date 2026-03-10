import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path="/Users/krishnashah/Desktop/customer_churn/data/telecom_churn.csv"):
	df = pd.read_csv(path)
	return df

def clean_data(df):
	df = df.dropna()
	return df

def encode_feature(df):
	le = LabelEncoder()
	for columns in df.columns:
		if df[columns].dtype == 'object':
			df[columns] = le.fit_transform(df[columns])
	return df

def split_features_target(df):
	X = df.drop('Churn', axis=1)
	y = df["Churn"]
	return X,y

def preprocess_data(path="/Users/krishnashah/Desktop/customer_churn/data/telecom_churn.csv"):
	df = load_data(path="/Users/krishnashah/Desktop/customer_churn/data/telecom_churn.csv")
	df = clean_data(df)
	df = encode_feature(df)
	X,y = split_features_target(df)
	return X,y
	