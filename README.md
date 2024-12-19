Website Chatbot using RAG (Retrieval-Augmented Generation)
This project demonstrates how to build a chatbot that can respond to queries using the content from a given website. It leverages the RAG (Retrieval-Augmented Generation) model from Hugging Face’s transformers library to retrieve relevant information and generate answers from the website’s content.

Features
Extracts text from a provided website URL.
Uses the RAG (Retrieval-Augmented Generation) model for generating responses based on the extracted content.
Provides a chatbot interface where users can ask questions related to the website's content.
Technologies Used
Python
BeautifulSoup (for web scraping)
Hugging Face transformers (for RAG model)
Torch (for model inference)
Requirements
To run this project, ensure you have the following Python libraries installed:

requests
beautifulsoup4
transformers
torch
You can install these dependencies by running the following command:
pip install requests beautifulsoup4 transformers torch
How to Use
Clone the repository:
git clone https://github.com/your-username/website-chatbot.git
cd website-chatbot
Run the script:
Execute the script to start the chatbot interface.
python chatbot.py
Provide a website URL:
The script will prompt you to enter a website URL. For example:
Enter the website URL: https://www.uchicago.edu/
Chat with the bot:
Once the content is fetched from the website, you can start interacting with the bot. Type your queries, and the bot will attempt to generate a relevant response from the website content.

To exit the chat, type exit.

How it Works
Extract Website Content: The extract_website_content() function fetches the website content and extracts all text from <p> tags (paragraphs).

Setup RAG Pipeline: The setup_rag_pipeline() function initializes the RAG model and tokenizer from Hugging Face’s transformers library.

Generate Response: The generate_response() function tokenizes the query and retrieves relevant documents using the website content as the document corpus. The response is generated using the RAG model.

Interactive Chat: The chatbot runs in an interactive loop where the user can ask questions about the website content and get responses based on the extracted text.


Notes
Ensure that the website you provide has publicly available text content. Websites with heavy JavaScript content might not work as expected.
The chatbot's responses are based solely on the content available on the provided website.
