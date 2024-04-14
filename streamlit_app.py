
import requests
import os
import PyPDF2
import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# URL of the PDF to be downloaded
url = "https://www.btg-bestellservice.de/pdf/80201000.pdf"

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Create a directory to store downloaded PDFs if it doesn't exist
    if not os.path.exists('pdf_files'):
        os.makedirs('pdf_files')
    
    # Extract the filename from the URL
    filename = url.split('/')[-1]

    # Specify the path to save the PDF file
    filepath = os.path.join('pdf_files', filename)

    # Write the content to a PDF file
    with open(filepath, 'wb') as pdf_file:
        pdf_file.write(response.content)
    
    print("PDF file downloaded successfully.")

    # Extract text from the PDF
    with open(filepath, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        
        print("Extracted text from PDF:")
        print(text)
      
        # Save the extracted text to a .txt file
        txt_filename = filename.split('.')[0] + '.txt'
        txt_filepath = os.path.join('pdf_files', txt_filename)
        
        print("Writing extracted text to:", txt_filepath)
        with open(txt_filepath, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)

def generate_response(text, openai_api_key, query_text):
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents([text])
    # Select embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # Create a vectorstore from documents
    db = Chroma.from_documents(texts, embeddings)
    # Create retriever interface
    retriever = db.as_retriever()
    # Create QA chain
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
    return qa.run(query_text)

# Page title
st.set_page_config(page_title='🦜🔗 Ask about the German Constitution')
st.title('🦜🔗 Ask about the German Constitution')

# Load text from the downloaded file
txt_filename = filename.split('.')[0] + '.txt'
txt_filepath = os.path.join('pdf_files', txt_filename)
with open(txt_filepath, 'r', encoding='utf-8') as txt_file:
    text_content = txt_file.read()

# Query text
query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.')

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not(query_text))
    submitted = st.form_submit_button('Submit', disabled=not(query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(text_content, openai_api_key, query_text)
            result.append(response)
            del openai_api_key

if len(result):
    st.info(result[0])

st.title('🎈 App Name')
st.write('Hello world!')
