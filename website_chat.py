import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
from flask import Flask, request, jsonify
from langchain.llms import OpenAI

# Initialize components
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained embedding model
dimension = 384  # Dimensionality of embeddings
faiss_index = faiss.IndexFlatL2(dimension)  # FAISS vector database
website_content = []  # Store scraped website content
metadata = []  # Metadata for context retrieval
llm = OpenAI(model="text-davinci-003")  # Initialize LLM

# Function to scrape website content
def scrape_website(url):
    """
    Crawl and scrape textual content from a target website.
    """
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.get_text().strip() for p in soup.find_all('p') if p.get_text().strip()]
        headers = [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3']) if h.get_text().strip()]
        return paragraphs + headers
    else:
        print(f"Failed to fetch {url}")
        return []

# Function to generate embeddings
def generate_embeddings(content):
    """
    Generate vector embeddings for the given content using the embedding model.
    """
    return embedding_model.encode(content)

# Function to query FAISS index
def query_faiss(query, top_k=5):
    """
    Perform a similarity search in the FAISS index.
    """
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(query_embedding, top_k)
    results = [{"content": website_content[i], "distance": distances[0][j]} for j, i in enumerate(indices[0])]
    return results

# Function to generate response using LLM
def generate_response(query, retrieved_chunks):
    """
    Generate a response using LLM based on the retrieved chunks and query.
    """
    context = "\n".join([chunk["content"] for chunk in retrieved_chunks])
    prompt = f"Based on the following context:\n{context}\nAnswer the question: {query}"
    return llm(prompt)

# Flask app for user interaction
app = Flask(__name__)

@app.route('/ingest', methods=['POST'])
def ingest_websites():
    """
    Endpoint to ingest websites and store embeddings in the FAISS index.
    """
    urls = request.json.get("urls", [])
    global website_content, metadata

    for url in urls:
        content = scrape_website(url)
        embeddings = generate_embeddings(content)
        faiss_index.add(embeddings)
        website_content.extend(content)
        metadata.extend([{"url": url, "content": text} for text in content])

    return jsonify({"message": "Websites ingested successfully", "total_content": len(website_content)})

@app.route('/query', methods=['POST'])
def handle_query():
    """
    Endpoint to handle user queries and return responses from the RAG pipeline.
    """
    data = request.json
    query = data.get("query", "")
    top_k = data.get("top_k", 5)

    # Retrieve relevant chunks
    retrieved_chunks = query_faiss(query, top_k)

    # Generate a response using LLM
    response = generate_response(query, retrieved_chunks)
    return jsonify({"response": response, "retrieved_chunks": retrieved_chunks})

# Main function to run the app
if __name__ == "__main__":
    app.run(port=5000)