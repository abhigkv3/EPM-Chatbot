import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# Page config
st.set_page_config(
    page_title="EPM Documentation Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Title
st.title("🤖 EPM Documentation Chatbot")
st.caption("351 pages EPM docs pe based AI Assistant")

# API Key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    groq_api_key = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_xxxxxxxxxx"
    )

# ✅ Casual messages handle karne ke liye
CASUAL_INPUTS = [
    "ok", "okay", "hi", "hello", "hey", "thanks",
    "thank you", "shukriya", "theek hai", "haan",
    "nahi", "bye", "good", "great", "nice", "hmm",
    "ok thanks", "got it", "understood", "sure",
    "k", "👍", "great thanks", "ty", "thx"
]

def is_casual_message(text):
    return text.lower().strip() in CASUAL_INPUTS

def get_casual_response(text):
    text = text.lower().strip()
    if text in ["hi", "hello", "hey"]:
        return "Hello! 👋 I'm your EPM Documentation Assistant. Feel free to ask any EPM related question!"
    elif text in ["thanks", "thank you", "shukriya", "ty", "thx", "ok thanks", "great thanks"]:
        return "Happy to help! 😊 Feel free to ask more EPM questions anytime!"
    elif text in ["bye"]:
        return "Goodbye! 👋 Come back anytime you have EPM questions!"
    elif text in ["ok", "okay", "theek hai", "got it", "understood", "sure", "k", "👍"]:
        return "Got it! Let me know if you have any more EPM questions 😊"
    elif text in ["good", "great", "nice"]:
        return "Glad to hear that! 😊 Any more EPM questions?"
    elif text in ["haan"]:
        return "Sure! Please go ahead with your EPM question 😊"
    elif text in ["nahi"]:
        return "Alright! Feel free to ask whenever you need help 😊"
    else:
        return "I'm your EPM Documentation Assistant. Please ask any EPM related question!"

# PDF load — cache
@st.cache_resource
def load_rag_system(api_key):
    with st.spinner("📄 PDF load ho rahi hai..."):
        loader = PyPDFLoader("EPM Documentation.pdf")
        pages = loader.load()

    with st.spinner("✂️ Chunks ban rahe hain..."):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(pages)

    with st.spinner("🔢 Vector Database ban raha hai..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        vectordb = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )

    retriever = vectordb.as_retriever(
        search_kwargs={"k": 7}
    )

    llm = ChatGroq(
        api_key=api_key,
        model="llama-3.3-70b-versatile"
    )

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
    retriever, llm, prompt = load_rag_system(groq_api_key)
    st.sidebar.success("✅ System Ready!")

    if user_input := st.chat_input("EPM ke baare mein kuch poochho..."):

        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # ✅ Casual check pehle
            if is_casual_message(user_input):
                answer = get_casual_response(user_input)
                st.markdown(answer)
            else:
                with st.spinner("Soch raha hoon..."):
    try:
        relevant_chunks = retriever.invoke(user_input)
        context = "\n\n".join([
            chunk.page_content for chunk in relevant_chunks
        ])
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "question": user_input
        })
    except Exception as e:
        if "rate_limit" in str(e).lower() or "429" in str(e):
            answer = "⚠️ Too many requests! Please wait 1-2 minutes and try again."
        else:
            answer = "⚠️ Something went wrong. Please try again!"
    st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
else:
    st.warning("⬅️ Pehle sidebar mein Groq API Key daalo!")
    st.info("Groq API Key yahan se lo: https://console.groq.com")