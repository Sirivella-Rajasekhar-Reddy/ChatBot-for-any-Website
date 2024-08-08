import validators, streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import os
load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")

st.set_page_config(page_title="ChatBot for any Website", page_icon="🤖")
st.title("🤖 ChatBot for any Website")

st.subheader("ChatBot")

with st.sidebar:
    groq_api_key=st.text_input("Enter Groq Api Key", type="password")
    web_url=st.text_input("Enter Website Url")

llm=ChatGroq(groq_api_key=groq_api_key, model="Gemma2-9b-It")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

system_template = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. if the question is not related to the "
    "retrieved context and If you don't know the answer, say that you "
    "don't know. keep the answer concise."
    "\n\n"
    "{context}"
)

chat_query=st.text_input("Enter Query to ChatBot")

if st.button("Answer"):
    try:
        if (not groq_api_key) and (not web_url):
            st.error("Please provide the Groq API key and Web Url to get started")
        elif not chat_query:
            st.error("Please provide a query to ChatBot")
        elif not validators.url(web_url):
            st.error("Please enter a valid Url. It can be website url")
        else:
            loader=WebBaseLoader(web_path=[web_url])
            docs=loader.load()
            text_splitter=RecursiveCharacterTextSplitter(separators="\n", chunk_size=1000, chunk_overlap=200)
            final_docs=text_splitter.split_documents(docs)
            vectore_storedb=FAISS.from_documents(final_docs, embeddings)
            retrieaver=vectore_storedb.as_retriever()
            prompt=PromptTemplate(input_variables=["context"], template=system_template)
            question_answer_chain=create_stuff_documents_chain(llm, prompt)
            rag_chain=create_retrieval_chain(retrieaver, question_answer_chain)
            response=rag_chain.invoke({"input":chat_query})
            st.success(response["answer"])
    except Exception as e:
        st.exception(f'exception : {e}')



    

