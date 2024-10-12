from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from ssl import create_default_context
import os
import warnings
import logging

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress logs from sentence-transformers and elastic_transport
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('elastic_transport').setLevel(logging.ERROR)

# Set up root logging level
logging.basicConfig(level=logging.ERROR)

# Load sensitive data from environment variables
ELASTIC_HOST = os.getenv('ELASTIC_HOST', "https://16b2d33a-7ea4-4b0d-b044-49d267194b9e.bc28ac43cf10402584b5f01db462d330.databases.appdomain.cloud:30176")
ELASTIC_USER = os.getenv('ELASTIC_USER', "ibm_cloud_61e8b85f_16de_450f_9874_859938612c2f")
ELASTIC_PW = os.getenv('ELASTIC_PW', "e813b7f861965411118fc9f67138062558c458727f0ca9c2357392a147c8bda9")
ELASTIC_CERT_FILE = os.getenv('ELASTIC_CERT_FILE', r"C:\Users\User\Desktop\testenv\cert.pem")  # SSL certificate

# Define the model ID for query vector generation
MODEL_ID = "sentence-transformers/paraphrase-MiniLM-L6-v2"

# Create an SSL context using the provided certificate file
context = create_default_context(cafile=ELASTIC_CERT_FILE)

# Initialize the Elasticsearch client
client = Elasticsearch(
    hosts=[ELASTIC_HOST],
    basic_auth=(ELASTIC_USER, ELASTIC_PW),
    ssl_context=context,
    verify_certs=True,
    request_timeout=60
)

# Load the transformer model
model = SentenceTransformer(MODEL_ID)

# Function to perform a kNN search and format results
def knn_search(query_text, k=5, num_candidates=1000):
    try:
        # Convert the query text to a vector (disable progress bar)
        query_vector = model.encode(query_text, show_progress_bar=False).tolist()

        # Perform kNN search
        response = client.search(
            index="en_index_data_v2",
            body={
                "size": k,
                "knn": {
                    "field": "vector",  # The field where your dense vector is stored
                    "query_vector": query_vector,
                    "k": k,
                    "num_candidates": num_candidates
                },
                "_source": ["text_field"]  # Retrieve only the 'text_field'
            }
        )

        hits = response['hits']['hits']
        if not hits:
            print("No similar documents found.")
        else:
            print(f"Found {len(hits)} similar document(s):\n")
            for hit in hits:
                text = hit['_source'].get('text_field', 'No text available')
                score = hit['_score']
                document_id = hit['_id']
                print(f"Document ID: {document_id}")
                print(f"Score: {score:.4f}")
                print(f"Text: {text}\n")
                print("-" * 50)  # Separator for better readability

        return hits
    except Exception as e:
        print(f"Error during kNN search: {e}")
        return None

# Main function to execute search
if __name__ == "__main__":
    query_text = "The second party has the right to leave work and terminate the contract"
    knn_search(query_text)
