import os

import pinecone

api_key = os.getenv("PINECONE_API_KEY")
environment = os.getenv("PINECONE_ENVIRONMENT")
print(api_key)
print(environment)

pinecone.init(api_key=api_key, environment=environment)

pinecone.create_index("quickstart", dimension=8, metric="euclidean")

indexes = pinecone.list_indexes()
print(indexes)

index = pinecone.Index("quickstart")
print(index)

index_stats = index.describe_index_stats()
print(index_stats)

index.upsert([
    ("A", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    ("B", [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
    ("C", [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
    ("D", [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]),
    ("E", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
])

result = index.query(vector=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], top_k=3, include_values=True)
print(result)

pinecone.delete_index("quickstart")
