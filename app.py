import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Page config
st.set_page_config(
    page_title="EPM Documentation Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Title
st.title("🤖 EPM Documentation Chatbot")
st.caption("340 pages EPM docs pe based AI Assistant")

# API Key input
groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    placeholder="gsk_xxxxxxxxxx"
)

# PDF load function — cache karo taki baar baar load na ho
@st.cache_resource
def load_rag_system(api_key):
    # PDF Load
    with st.spinner("📄 PDF load ho rahi hai..."):
        loader = PyPDFLoader("EPM Documentation.pdf")
        pages = loader.load()

    # Chunks
    with st.spinner("✂️ Chunks ban rahe hain..."):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(pages)

    # Vector DB
    with st.spinner("🔢 Vector Database ban raha hai..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )

    # Retriever
    retriever = vectordb.as_retriever(
        search_kwargs={"k": 7}
    )

    # LLM
    llm = ChatGroq(
        api_key=api_key,
        model="llama-3.3-70b-versatile"
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template("""
You are an EPM documentation expert assistant.
Answer in detail based on the context provided.
If the context has partial information, use it fully.
If not in context at all, say "Documentation mein nahi mila."
Always answer in simple English with steps if applicable.

Context: {context}

Question: {question}

Detailed Answer:
""")

    return retriever, llm, prompt

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat history display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat
if groq_api_key:
    # RAG system load karo
    retriever, llm, prompt = load_rag_system(groq_api_key)
    st.sidebar.success("✅ System Ready!")

    # User input
    if user_input := st.chat_input("EPM ke baare mein kuch poochho..."):

        # User message show karo
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        with st.chat_message("user"):
            st.markdown(user_input)

        # AI response
        with st.chat_message("assistant"):
            with st.spinner("Soch raha hoon..."):
                # Relevant chunks dhundho
                relevant_chunks = retriever.invoke(user_input)
                context = "\n\n".join([
                    chunk.page_content for chunk in relevant_chunks
                ])

                # Answer lo
                chain = prompt | llm | StrOutputParser()
                answer = chain.invoke({
                    "context": context,
                    "question": user_input
                })

                st.markdown(answer)

        # AI message save karo
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
else:
    st.warning("⬅️ Pehle sidebar mein Groq API Key daalo!")
    st.info("Groq API Key yahan se lo: https://console.groq.com")