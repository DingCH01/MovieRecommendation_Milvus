# -*- coding: utf-8 -*-

"""
# @Author   : Gary Ding
# @Time     : 2024-03-25 12:20
# @File     : movie_milvus.py
# @Project  : I3
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
data_path = "test_user_movie_rating.csv"  # Same data format as the movie recommendation service, but in a smaller scale
data = pd.read_csv(data_path)

# Prepare for feature engineering and extract vectors
data['MovieText'] = data['MovieID'].apply(lambda x: x.replace('+', ' '))

# Use TF-IDF to fit the size of the vectors
tfidf = TfidfVectorizer(max_features=80)
X = tfidf.fit_transform(data['MovieText']).toarray()
y = data['Rating'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Complete the traditional pipeline of machine learning
model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model MSE: {mse}")

# Introducing Milvus
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# Connect to Milvus, should be installed already
connections.connect("default", host='localhost', port='19530')

# Define the collection and the vector dim
collection_name = 'movie_vectors_02'
dim = 80

fields = [
    FieldSchema(name="movie_id", dtype=DataType.INT64, is_primary=True),  # Primary key
    FieldSchema(name="TFIDF_Vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
]

schema = CollectionSchema(fields, description="Movie TFIDF Vectors")
collection = Collection(collection_name, schema)


# Insert vectors into Milvus collections
movie_ids = list(range(len(X)))
mr = collection.insert([movie_ids, X.tolist()])

print(f"Inserted {len(mr.primary_keys)} vectors into Milvus.")

# Load the collection into memory before searching
collection.load()

# Searching 5 most similar movies in Milvus
results = collection.search(X[:1].tolist(), "TFIDF_Vector", param={"metric_type": "L2", "params": {"nprobe": 10}},
                            limit=5)

for hits in results:
    for hit in hits:
        print(f"Hit: {hit}, Distance: {hit.distance}")
