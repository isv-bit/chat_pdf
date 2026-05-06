import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# ------------------ CONFIGURACIÓN VISUAL ------------------
st.set_page_config(
    page_title="Analizador RAG",
    page_icon="📄",
    layout="wide"
)

# CSS personalizado
st.markdown("""
    <style>
    body {
        background-color: #f5f7fa;
    }
    .main-title {
        font-size: 40px;
        font-weight: bold;
        color: ##2c5c10;
    }
    .subtitle {
        font-size: 18px;
        color: #5D6D7E;
    }
    .box {
        background-color: #EBF5FB;
        padding: 15px;
        border-radius: 10px;
        border-left: 6px solid #3498DB;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown('<p class="main-title">📊 Analizador de Familias Viajeras (RAG)</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analiza documentos sobre familias nómadas o viajeras usando inteligencia artificial</p>', unsafe_allow_html=True)

st.write("Versión de Python:", platform.python_version())

# ------------------ IMAGEN ------------------
try:
    image = Image.open('familia.jpg')
    st.image(image, width=300)  # más pequeña y tipo banner
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# ------------------ INDICACIONES ------------------
st.markdown("""
<div class="box">
📌 <b>Indicaciones:</b><br>
Sube un archivo PDF que trate sobre <b>familias viajeras o nómadas</b>.<br>
Puede incluir temas como:
<ul>
<li>Estilos de vida nómadas</li>
<li>Viajes en familia</li>
<li>Educación en movimiento</li>
<li>Experiencias culturales</li>
</ul>
Luego escribe una pregunta y el sistema analizará el documento para responderte.
</div>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.subheader("ℹ️ Información")
    st.write("Este agente analiza documentos PDF utilizando RAG (Recuperación + Generación).")

# ------------------ API KEY ------------------
ke = st.text_input('🔑 Ingresa tu Clave de OpenAI', type="password")

if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# ------------------ SUBIR PDF ------------------
pdf = st.file_uploader("📄 Carga el archivo PDF", type="pdf")

# ------------------ PROCESAMIENTO ------------------
if pdf is not None and ke:
    try:
        # Leer PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        st.info(f"📊 Texto extraído: {len(text)} caracteres")

        # Dividir texto
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"Documento dividido en {len(chunks)} fragmentos")

        # Embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Pregunta
        st.subheader("❓ Haz una pregunta sobre el documento")
        user_question = st.text_area("", placeholder="Ej: ¿Cómo viven las familias nómadas?")

        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI(
                temperature=0,
                model_name="gpt-4o-mini-2024-07-18"
            )

            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            st.markdown("### 🧠 Respuesta:")
            st.markdown(response)

    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

else:
    st.info("📂 Sube un archivo PDF para comenzar")
