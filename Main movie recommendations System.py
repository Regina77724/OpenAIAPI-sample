import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# -----------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# -----------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate environment
if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("Missing API keys. Check your .env file.")

# -----------------------------
# 2. INIT OPENAI CLIENT
# -----------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# 3. READ MOVIE DATASET
# -----------------------------
data = pd.read_csv("movies.csv", dtype=str)
small_dataset = data[["original_title", "overview"]].head(200)  # limit for testing

# Clean missing text
small_dataset = small_dataset.fillna("")

# ------------------------------------
# 4. CREATE EMBEDDINGS FOR EACH MOVIE
# ------------------------------------
def embed_text(text):
    """Return 3072-dimensional OpenAI embedding."""
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding  # vector list

print("Creating embeddings...")

movie_embeddings = []
for i in range(len(small_dataset)):
    content = small_dataset.iloc[i]["overview"]
    movie_embeddings.append(embed_text(content))

print("Embeddings created for dataset!")

# -----------------------------------
# 5. INIT PINECONE + CREATE AN INDEX
# -----------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "movie"

# Create index only once
if index_name not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=3072,  # MUST match OpenAI embedding size
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("Index created!")

# Connect to existing index
index = pc.Index(index_name)

# ----------------------------
# 6. UPSERT VECTORS TO PINECONE
# ----------------------------
print("Uploading vectors to Pinecone...")

for i in range(len(small_dataset)):
    vid = str(i)  # unique ID
    vec = movie_embeddings[i]
    meta = {
        "title": small_dataset.iloc[i]["original_title"],
        "overview": small_dataset.iloc[i]["overview"]
    }

    index.upsert(
        vectors=[{
            "id": vid,
            "values": vec,
            "metadata": meta
        }]
    )

print("Upload completed!")

# -----------------------------------
# 7. USER QUERY → EMBEDDING → SEARCH
# -----------------------------------
user_query = input("\nEnter your movie description: ")

# Create embedding for user input
uq_vector = client.embeddings.create(
    model="text-embedding-3-large",
    input=user_query
).data[0].embedding

# Query Pinecone - use keyword args
result = index.query(
    vector=uq_vector,
    top_k=10,
    include_metadata=True
)

# -----------------------------
# 8. SHOW SEARCH RESULTS
# -----------------------------
print("\nTop Recommended Movies:\n")

for match in result["matches"]:
    title = match["metadata"]["title"]
    score = match["score"]
    print(f"{title}  (score: {round(score, 4)})")
