import os
import time
import shutil
import requests
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# === Config ===
PDF_STORAGE_PATH = 'documents'
VECTOR_DB_DIR = 'vector_indexes'
PROMPT_TEMPLATE = """
You are an expert research assistant. Use the context below — especially any tables — to answer the query. 
If the answer is not clearly stated, say you don't know.

Query: {user_query}
Context:
{document_context}

Answer:
"""

# Vérifie si Ollama est disponible
def wait_for_ollama_ready(host="http://localhost:11434", timeout=60):
    import streamlit as st
    import time
    import requests

    st.info("⏳ Connexion à Ollama...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            res = requests.get(f"{host}/api/tags")
            if res.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)

    st.error("❌ Ollama n’a pas répondu dans les délais.")
    return False


@st.cache_resource
def get_models(model_name):
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    if not wait_for_ollama_ready(ollama_host, timeout=120):
        st.stop()

    embedding_model = OllamaEmbeddings(model=model_name, base_url=ollama_host)
    language_model = OllamaLLM(model=model_name, temperature=0.0, base_url=ollama_host)

    return embedding_model, language_model


def load_all_pdfs_from_directory(directory_path):
    all_docs = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            path = os.path.join(directory_path, filename)
            loader = PDFPlumberLoader(path)
            pages = loader.load()
            text = "\n\n".join([p.page_content for p in pages])
            doc = Document(page_content=text, metadata={"source": path, "filename": filename, "page_count": len(pages)})
            all_docs.append(doc)
    return all_docs

def chunk_documents(raw_documents):
    if not raw_documents:
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150, add_start_index=True)
    return splitter.split_documents(raw_documents)

def create_or_load_vectorstore(chunks, embedding_model, db_type, persist_path, force_rebuild=False):
    if force_rebuild and os.path.exists(persist_path):
        shutil.rmtree(persist_path)

    if db_type == "FAISS":
        if force_rebuild:
            if not chunks:
                return None
            vectorstore = FAISS.from_documents(chunks, embedding_model)
            os.makedirs(persist_path, exist_ok=True)
            vectorstore.save_local(persist_path)
            return vectorstore
        elif os.path.exists(persist_path):
            return FAISS.load_local(persist_path, embedding_model, allow_dangerous_deserialization=True)
    else:
        if force_rebuild:
            return Chroma.from_documents(chunks, embedding_model, persist_directory=persist_path)
        elif os.path.exists(persist_path):
            return Chroma(persist_directory=persist_path, embedding_function=embedding_model)
    return None

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_answer(user_query, retriever, language_model):
    if not retriever:
        return "Aucun index disponible.", []

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    try:
        docs = retriever.get_relevant_documents(user_query)
        if not docs:
            return "Aucun document pertinent trouvé.", []
        context = format_docs(docs)
        chain_input = {"document_context": context, "user_query": user_query}
        chain = prompt | language_model | StrOutputParser()
        answer = chain.invoke(chain_input)
        return answer, docs
    except Exception as e:
        return f"Erreur : {str(e)}", []

# === Streamlit UI ===
st.set_page_config(page_title="RAG PDF Assistant", layout="wide")
st.title("📄🔎 Assistant de recherche RAG PDF")

# Sidebar
st.sidebar.header("🧠 Choix des paramètres")
model_choice = st.sidebar.selectbox("Modèle :", [
    "mistral", 
    "llama3.1",
    "llama3.2", 
    "deepseek-r1:7b",
    "deepseek-r1:8b", 
    "wizardlm2", 
    "llama3-chatqa"
])
vectorstore_choice = st.sidebar.selectbox("Base vectorielle :", ["FAISS", "Chroma"])
embedding_model, language_model = get_models(model_choice)

index_dir_name = f"{vectorstore_choice.lower()}_{model_choice.replace(':', '_')}"
index_path = os.path.join(VECTOR_DB_DIR, index_dir_name)

# Fichiers PDF
uploaded_files = st.sidebar.file_uploader("📤 Ajouter des PDF", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    os.makedirs(PDF_STORAGE_PATH, exist_ok=True)
    for uploaded_file in uploaded_files:
        with open(os.path.join(PDF_STORAGE_PATH, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.read())
    st.sidebar.success("Fichiers importés.")

# Réindexation
if st.sidebar.button("🔁 Recharger et réindexer les PDF"):
    with st.spinner("Indexation en cours..."):
        raw_docs = load_all_pdfs_from_directory(PDF_STORAGE_PATH)
        chunks = chunk_documents(raw_docs)
        vectorstore = create_or_load_vectorstore(chunks, embedding_model, vectorstore_choice, index_path, force_rebuild=True)
        st.sidebar.success(f"{len(chunks)} chunks indexés.")
else:
    vectorstore = create_or_load_vectorstore([], embedding_model, vectorstore_choice, index_path, force_rebuild=False)

retriever = vectorstore.as_retriever() if vectorstore else None

# Question utilisateur
st.subheader("💬 Posez une question")
query = st.text_input("Votre question ici...")
with st.expander("⚙️ Paramètres avancés"):
    k_docs = st.slider("Nombre de documents à récupérer", min_value=1, max_value=10, value=4)
    if retriever:
        retriever.search_kwargs = {"k": k_docs}

if retriever:
    retriever.search_kwargs = {"k": k_docs}

if st.button("🔍 Rechercher"):
    if not query.strip():
        st.warning("Veuillez entrer une question.")
    else:
        with st.spinner("Recherche en cours..."):
            answer, used_docs = generate_answer(query, retriever, language_model)
            st.subheader("🧠 Réponse")
            st.write(answer)
            with st.expander("📄 Documents utilisés"):
                for i, doc in enumerate(used_docs, 1):
                    st.markdown(f"**{i}. {doc.metadata.get('filename')}**")
                    st.text_area(f"Contenu du chunk {i}", doc.page_content, height=200, key=f"chunk_{i}")

# === Questions prédéfinies ===
st.subheader("💡 Questions prédéfinies")
predefined_questions = [
    "Qui détient la plus grande valeur dans la section 'Speed' ?",
    "Donne moi les informations de Bob?",
    "Combien de participants sont inscrits dans le programme 'Blind' ?",
    "How many tables you have found?",
    "Give me all Disability Categories ?",
    "Give me Info about Electromagnets-Increasing Coils ?"
]

if st.button("▶️ Lancer toutes les questions prédéfinies"):
    if not retriever:
        st.error("Veuillez d'abord indexer vos documents.")
    else:
        for question in predefined_questions:
            st.markdown(f"#### 🔎 Question : {question}")
            answer, _ = generate_answer(question, retriever, language_model)
            st.write(f"**Réponse :** {answer}")