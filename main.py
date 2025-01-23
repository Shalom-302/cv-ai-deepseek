import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai


# Charger les variables d'environnement
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

api_key1 = os.getenv("GOOGLE_API_KEY")

# Configurer le client DeepSeek
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1"  # Endpoint DeepSeek
)

genai.configure(api_key=api_key1)



# Fonction pour extraire le texte d'un PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Fonction pour diviser le texte en chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Fonction pour créer un vector store avec FAISS
def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Fonction pour analyser le texte avec DeepSeek
def analyze_text_with_deepseek(text):
    messages = [
        {
            "role": "system",
            "content": """Extrayez les informations suivantes du CV:
            - ÉDUCATION (liste des diplômes, établissements et années)
            - EXPÉRIENCES (postes, entreprises, dates, descriptions)
            - COMPÉTENCES (techniques et soft skills)
            - LANGUES (listées avec niveaux)
            - CERTIFICATIONS
            - CONTACT (email, téléphone, LinkedIn)"""
        },
        {
            "role": "user",
            "content": text
        }
    ]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        max_tokens=4000,
        temperature=0.5
    )

    return response.choices[0].message.content

# Fonction pour analyser une image avec Gemini
def analyze_image_with_gemini(image):
    model = genai.GenerativeModel('gemini-1.5-pro')
    prompt = """
    Extrais les informations suivantes de ce CV :
    - ÉDUCATION (liste des diplômes, établissements et années)
    - EXPÉRIENCES (liste des postes, entreprises, dates et descriptions)
    - COMPÉTENCES (liste des compétences techniques et soft skills)
    - LANGUES (liste des langues parlées et niveaux)
    - CERTIFICATIONS (liste des certifications obtenues)
    - CONTACT (email, téléphone, LinkedIn, etc.)
    """
    response = model.generate_content([prompt, image])
    return response.text

# Interface Streamlit
def main():
    st.set_page_config("Analyseur de CV")
    st.header("Analyseur de CV avec DeepSeek")

    # Upload de fichier (PDF ou image)
    uploaded_file = st.file_uploader("Téléchargez un CV (PDF ou Image)", type=["pdf", "jpg", "jpeg", "png"])

    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            st.write("Fichier PDF détecté.")
            # Extraire le texte du PDF
            raw_text = get_pdf_text([uploaded_file])
            st.subheader("Texte extrait du PDF :")

            # Analyser le texte avec DeepSeek
            st.subheader("Analyse du CV :")
            with st.spinner("Analyse en cours..."):
                analysis_result = analyze_text_with_deepseek(raw_text)
                st.write(analysis_result)

        else:
            st.write("Fichier image détecté.")
            # Ouvrir l'image
            image = Image.open(uploaded_file)
            st.image(image, caption='Image téléchargée', use_container_width=True)

            # Analyser l'image avec DeepSeek
            st.subheader("Analyse du CV :")
            with st.spinner("Analyse en cours..."):
                analysis_result = analyze_image_with_gemini(image)
                st.write(analysis_result)

# Point d'entrée
if __name__ == "__main__":
    main()