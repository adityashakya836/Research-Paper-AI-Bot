# import streamlit as st
# import google.generativeai as genai 
# from PyPDF2 import PdfReader
# import json
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# GOOGLE_API_KEY = st.secrets['GOOGLE_API_KEY']

# # Reading the text from the pdfs
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# # Splitting the text into chunks
# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=10000,
#         chunk_overlap=1000
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks

# # Convert the chunks into vectors
# def get_vector_store(text_chunks):
#     embeddings = HuggingFaceBgeEmbeddings()
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local('faiss_index')

# # Make a Conversational Chain
# def get_conversational_chain():
#     prompt_template = """
#         You are an expert in analyzing and understanding research papers. Your role is to assist in writing or completing sections of an incomplete research paper with clarity, precision, and technical accuracy. You excel at identifying key aspects of research and explaining advanced concepts, including mathematical formulations and algorithms.

#         Your tasks include drafting and completing the following sections based on the provided context:

#         1. **Abstract** (Present Tense):  
#         - Provide a 2-3 line introduction to the problem addressed in the research.  
#         - Highlight the limitations of existing methods or approaches.  
#         - Present the proposed method and its advantages, emphasizing its novelty or superiority.

#         2. **Conclusion** (Past Tense):  
#         - Summarize the key findings of the research.  
#         - Suggest potential future work or improvements in the field.

#         3. **Scope of the Research**:  
#         - Clearly state the research objectives, its scope, and its relevance to the domain.

#         4. **Introduction and Research Gap**:  
#         - Provide the background and motivation for the study.  
#         - Identify the specific research gap the study intends to address.

#         5. **Limitations**:  
#         - Discuss constraints, challenges, or shortcomings faced in the research.  
#         - Explain their impact on the findings or outcomes.

#         6. **Methodology**:  
#         - Detail the methods or approaches used, including relevant mathematical formulations, algorithms, or models.  
#         - Ensure mathematical explanations are clear, accurate, and logically structured.

#         7. **Evaluation**:  
#         - Explain how the results were validated using metrics, experiments, or comparative analyses.  
#         - Emphasize the performance of the proposed method with evidence from the research.

#         Additional Expertise:  
#         - You have a deep understanding of advanced mathematics and algorithms.  
#         - When asked, you can generate relevant code snippets in any programming language.  
#         - If information is unavailable in the context, respond concisely with: *"The answer is not available in the provided context."*

#         **Context**:  
#         {context}

#         **Question**:  
#         {question}

#         **Answer**:
#     """

#     model = ChatGoogleGenerativeAI(model = 'gemini-1.5-pro', temperature = 0.7, google_api_key = GOOGLE_API_KEY)
#     prompt = PromptTemplate(template = prompt_template, input_variables = ['context', 'question'])
#     chain = load_qa_chain(model, chain_type='stuff',prompt = prompt)
#     return chain

# # Take the user input
# def user_input(user_question):
#     embeddings = HuggingFaceBgeEmbeddings()
#     new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain(
#         {
#             "input_documents": docs,
#             'question': user_question
#         },
#         return_only_outputs=True
#     )

#     # Print message
#     st.write("Reply: ", response['output_text'])

# # Making an interface
# def main():
#     st.set_page_config("Scholar Bot")
#     st.header("Scholar Bot")

#     user_question = st.text_input("Ask a question from the PDF Files")
#     if user_question:
#         user_input(user_question)
    
#     with st.sidebar:
#         st.title('Menu:')
#         pdf_docs = st.file_uploader("Upload you PDF Files and Click on Submit & Process Button", type='pdf',
#             accept_multiple_files=True)

#         if st.button('Submit & Process'):
#             with st.spinner('Processing...'):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Done and You can ask query.")

# if __name__ == '__main__':
#     main()



# import streamlit as st
# import google.generativeai as genai
# from PyPDF2 import PdfReader
# import json
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# GOOGLE_API_KEY = st.secrets['GOOGLE_API_KEY']

# # Store conversation history
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []

# # Read text from PDFs
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text() if page.extract_text() else ""
#     return text

# # Split text into chunks
# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=10000, chunk_overlap=1000
#     )
#     return text_splitter.split_text(text)

# # Convert text chunks to vectors
# def get_vector_store(text_chunks):
#     embeddings = HuggingFaceBgeEmbeddings()
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local('faiss_index')

# # Define Conversational Chain
# def get_conversational_chain():
#     prompt_template = """
#         You are an expert in analyzing research papers. Answer based on the given context.
#         If context is unavailable, reply: *"The answer is not available in the provided context."*
        
#         **Context**:  
#         {context}  
#         **Question**:  
#         {question}  
#         **Answer**:
#     """
#     model = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0.7, google_api_key=GOOGLE_API_KEY)
#     prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
#     return load_qa_chain(model, chain_type='stuff', prompt=prompt)

# # Process user input
# def user_input(user_question):
#     embeddings = HuggingFaceBgeEmbeddings()
#     new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain({'input_documents': docs, 'question': user_question}, return_only_outputs=True)
#     answer = response['output_text']
    
#     st.session_state.chat_history.append((user_question, answer))
#     st.write("Reply:", answer)

# # Summarize uploaded PDFs
# def summarize_pdfs():
#     embeddings = HuggingFaceBgeEmbeddings()
#     new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
#     summary_query = "Summarize the key points of the research papers."
#     docs = new_db.similarity_search(summary_query)
#     chain = get_conversational_chain()
#     response = chain({'input_documents': docs, 'question': summary_query}, return_only_outputs=True)
#     st.write("Summary:", response['output_text'])

# # Streamlit App UI
# def main():
#     st.set_page_config("Scholar Bot")
#     st.header("📚 Scholar Bot - AI Research Assistant")
    
#     user_question = st.text_input("Ask a question about the uploaded PDFs:")
#     if user_question:
#         user_input(user_question)
    
#     with st.sidebar:
#         st.title("📂 Upload Research Papers")
#         pdf_docs = st.file_uploader("Upload PDF Files", type='pdf', accept_multiple_files=True)
        
#         if st.button("Process PDFs"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Processing complete. You can now ask questions!")
        
#         if st.button("Summarize PDFs"):
#             summarize_pdfs()
    
#     st.subheader("💬 Chat History")
#     for question, answer in st.session_state.chat_history:
#         st.write(f"**Q:** {question}")
#         st.write(f"**A:** {answer}")

# if __name__ == '__main__':
#     main()


import streamlit as st
import google.generativeai as genai 
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Set API Key
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# Function to read PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Function to create vector store
def get_vector_store(text_chunks):
    embeddings = HuggingFaceBgeEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create the conversational chain
def get_conversational_chain():
    prompt_template = """
    You are an expert in research paper analysis. Use the provided context to answer questions precisely.
    
    **Context:**
    {context}
    
    **Question:**
    {question}
    
    **Answer:**
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, google_api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return model, prompt

# Function to process user input
def user_input(user_question, chat_history):
    embeddings = HuggingFaceBgeEmbeddings()
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    model, prompt = get_conversational_chain()
    response = model.generate(prompt.format(context=docs, question=user_question))
    chat_history.append((user_question, response))
    return response

# Streamlit ChatGPT-like UI
def main():
    st.set_page_config(page_title="Scholar Bot", layout="wide")
    st.title("📚 Scholar Bot: AI Research Assistant")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat interface
    user_question = st.chat_input("Ask a question from the PDF files...")
    if user_question:
        response = user_input(user_question, st.session_state.chat_history)
        st.session_state.chat_history.append((user_question, response))
    
    # Display chat history
    for query, reply in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            st.write(reply)
    
    # Sidebar for file upload
    with st.sidebar:
        st.title("📂 Upload PDFs")
        pdf_docs = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)
        if st.button("📑 Process PDFs"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ PDF processed! You can now ask questions.")

if __name__ == "__main__":
    main()
