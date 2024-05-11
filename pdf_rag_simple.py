from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import streamlit as st
import os
from langchain_groq import ChatGroq


os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

chat = ChatGroq(temperature=1, model_name="llama3-70b-8192")

uploaded_file = st.file_uploader('upload your pdf here',type='pdf')
input = st.text_input('what do you wanna know about the text?')
key = st.button('click to start')

if uploaded_file and key:
    os.makedirs("temp", exist_ok=True)
    with open(os.path.join("temp", uploaded_file.name), "wb") as file:
        file.write(uploaded_file.getbuffer())

    loader = UnstructuredPDFLoader(os.path.join("temp", uploaded_file.name))
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    splits = splitter.split_documents(data)

    vector_db = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(),
        collection_name="local-rag")

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate three
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        chat,
        prompt=QUERY_PROMPT
    )

    WRITER_SYSTEM_PROMPT = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."  # noqa: E501
    # Report prompts from https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/master/prompts.py
    RESEARCH_REPORT_TEMPLATE = """Information:
    --------
    {text}
    --------
    Using the above information, answer the following question or topic: "{question}" in a short manner-- \
    The answer should focus on the answer to the question, should be well structured, informative, \
    in depth, with facts and numbers if available and a minimum of 150 words and a maximum of 300 words.
    You should strive to write the report using all relevant and necessary information provided.
    You must write the report with markdown syntax.
    You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
    You must write the sources used in the context. if any article is used, mentioned in the end.
    Please do your best, this is very important to my career."""  # noqa: E501

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", WRITER_SYSTEM_PROMPT),
            ("user", RESEARCH_REPORT_TEMPLATE),
        ]
    )

    chain = (
            {"text": retriever, "question": RunnablePassthrough()}
            | prompt
            | chat
            | StrOutputParser()
    )

    answer = chain.invoke(
        {
            "question": input
        }
    )
    st.write(answer)
    vector_db.delete_collection()
