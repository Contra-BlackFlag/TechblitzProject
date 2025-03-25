import pandas as pd
import numpy as np
import os
import chromadb  # Vector DB
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from flask import Flask, request, render_template, jsonify

# Initialize Flask app
app = Flask(__name__)

# Define file path
file_path = r"C:\Hackathon\yahoo_data.xlsx"

# Check if file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File '{file_path}' not found. Please check the path.")

# Load dataset
df = pd.read_excel(file_path)

# Ensure required columns exist
required_columns = {'Date', 'Close'}
df.columns = df.columns.str.strip()  # Remove whitespace issues
column_mapping = {"Close*": "Close"}  # Handle column name variations
df.rename(columns=column_mapping, inplace=True)

if not required_columns.issubset(df.columns):
    raise KeyError(f"Dataset must contain {required_columns} columns. Found: {df.columns}")

# Convert Date to datetime and sort
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.sort_index()

# Create financial trend summaries
df['Trend'] = df['Close'].diff().apply(lambda x: "Up" if x > 0 else "Down" if x < 0 else "Stable")
df['Summary'] = df.index.astype(str) + ": Close Price " + df['Close'].astype(str) + " (" + df['Trend'] + ")"

# Load Sentence Transformer Model
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Convert text to embeddings
summaries = df['Summary'].tolist()
embeddings = encoder.encode(summaries)

# Initialize ChromaDB
db = chromadb.PersistentClient(path="./vector_db")
collection = db.get_or_create_collection(name="financial_data")

# Store embeddings in Vector DB
for i, (summary, embedding) in enumerate(zip(summaries, embeddings)):
    collection.add(
        documents=[summary],
        metadatas=[{"index": i}],
        ids=[str(i)]
    )

print(f"Stored {len(summaries)} financial data points in the vector database.")

# Setup Gemini API
genai.configure(api_key="YOUR API KEY HERE")

# List available models
def list_available_models():
    try:
        models = genai.list_models()
        available_model_names = [model.name for model in models]
        print("\n✅ Available Gemini Models:")
        for model in available_model_names:
            print(f"- {model}")
        return available_model_names
    except Exception as e:
        print(f"❌ Error fetching models: {str(e)}")
        return []

# Choose the best available Gemini model
available_models = list_available_models()
if available_models:
    model_name = next(
        (m for m in available_models if "gemini-2.0-flash" in m), 
        available_models[0]  # Default to the first model if no match
    )
else:
    raise RuntimeError("❌ No available models found. Check API key and connection.")

# Initialize model
model = genai.GenerativeModel(model_name)
print(f"\n✅ Using model: {model_name}")

def query_rag(prompt):
    """Retrieve relevant financial data & use Gemini API for forecasting."""
    # Retrieve relevant past trends
    query_embedding = encoder.encode([prompt])[0]  # Ensure proper format
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    if not results['documents']:
        return "No relevant financial data found."

    retrieved_context = "\n".join(results['documents'][0])  # Extract retrieved docs
    print("Retrieved Context:\n", retrieved_context)

    # Use Gemini API to generate insights with stock market dataset context
    full_prompt = f"""
    You are an AI financial analyst. The dataset consists of daily stock market data for multiple assets, including equities, ETFs, and indexes, spanning from April 1, 2018, to March 31, 2023.
    
    Based on historical stock trends:
    {retrieved_context}
    
    Provide an analysis or predict future stock trends.
    """

    try:
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error in AI response: {str(e)}"

# Flask Routes
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_query = request.form["query"]
    forecast = query_rag(user_query)
    return jsonify({"response": forecast})

if __name__ == "__main__":
    app.run(debug=True)
