import requests
from bs4 import BeautifulSoup
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import torch

# Function to extract text from a website
def extract_website_content(url):
    # Fetch the webpage
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to retrieve the website: {url}")
        return ""
    
    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract all text from the paragraphs (<p> tags)
    paragraphs = soup.find_all('p')
    text = " ".join([para.get_text() for para in paragraphs])
    
    return text

# Setup the RAG model pipeline
def setup_rag_pipeline():
    # Load pre-trained RAG model and tokenizer
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    retriever = RagRetriever.from_pretrained("facebook/rag-token-nq")
    model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq")
    
    return tokenizer, retriever, model

# Function to generate responses using the RAG pipeline
def generate_response(query, tokenizer, retriever, model, website_content):
    # Tokenize the query
    inputs = tokenizer(query, return_tensors="pt")
    
    # Retrieve relevant documents based on the website content
    # Note: This example assumes the website content is passed as the document corpus
    input_ids = inputs['input_ids']
    doc_scores, docs = retriever(input_ids, website_content)
    
    # Generate a response using the model
    generated_ids = model.generate(input_ids=input_ids, context_input_ids=docs['input_ids'], max_length=200)
    
    # Decode the generated response
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return response

# Main function to handle the chatbot interface
def chat_with_website(url):
    # Step 1: Extract content from the website
    website_content = extract_website_content(url)
    
    if not website_content:
        print("No content available from the website.")
        return
    
    # Step 2: Set up the RAG model pipeline
    tokenizer, retriever, model = setup_rag_pipeline()
    
    print("Chatbot is ready! Ask something about the website (type 'exit' to stop):")
    
    while True:
        # Step 3: Get the user input
        query = input("You: ")
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Step 4: Generate response from the website content
        response = generate_response(query, tokenizer, retriever, model, website_content)
        print(f"Bot: {response}")

# Example usage: Provide a website URL
if __name__ == "__main__":
    url = input("https://www.uchicago.edu/")
    chat_with_website(url)
