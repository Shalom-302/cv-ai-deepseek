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

# Configurer les clients API
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
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
def analyze_text_with_deepseek(text, output_format):
    # Adaptation du prompt selon le format de sortie
    if output_format == "JSON":
        system_content = """Extrayez les informations suivantes du CV puis formattez-les en JSON de façon à les intégrer dans une base de données :"""
    else:
        system_content = """Extrayez les informations suivantes du CV et présentez-les sous forme de texte brut structuré :"""
    
    full_prompt = f"""
    {system_content}
    - ÉDUCATION (liste des diplômes, établissements et années)
    - EXPÉRIENCES (postes, entreprises, dates, descriptions)
    - COMPÉTENCES (techniques et soft skills)
    - LANGUES (listées avec niveaux)
    - CERTIFICATIONS
    - CONTACT (email, téléphone, LinkedIn)
    """

    messages = [
        {"role": "system", "content": full_prompt},
        {"role": "user", "content": text}
    ]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        max_tokens=4000,
        temperature=0.5
    )

    return response.choices[0].message.content

# Fonction pour analyser une image avec Gemini
def analyze_image_with_gemini(image, output_format):
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    # Adaptation du prompt selon le format de sortie
    if output_format == "JSON":
        prompt = """Extrayez les informations suivantes du CV puis formattez-les en JSON :"""
    else:
        prompt = """Extrayez les informations suivantes du CV et présentez-les sous forme de texte brut structuré :"""
    
    full_prompt = f"""
    {prompt}
    - ÉDUCATION (liste des diplômes, établissements et années)
    - EXPÉRIENCES (liste des postes, entreprises, dates et descriptions)
    - COMPÉTENCES (liste des compétences techniques et soft skills)
    - LANGUES (liste des langues parlées et niveaux)
    - CERTIFICATIONS (liste des certifications obtenues)
    - CONTACT (email, téléphone, LinkedIn, etc.)
    """

    response = model.generate_content([full_prompt, image])
    return response.text

def main():
    st.set_page_config(page_title="Analyseur de CV", page_icon="📄")
    st.header("Analyseur de CV avec DeepSeek & Gemini")

    # Sélection du type de fichier
    file_type = st.radio(
        "Type de fichier à analyser :",
        options=["PDF", "Image"],
        index=0
    )

    # Sélection du format de sortie
    output_format = st.radio(
        "Format de sortie :",
        options=["JSON", "Texte brut"],
        index=0
    )

    # Gestion de l'upload et du traitement
    if file_type == "PDF":
        uploaded_file = st.file_uploader("Téléchargez votre CV (PDF)", type=["pdf"])
        if uploaded_file:
            with st.spinner("Extraction du texte..."):
                raw_text = get_pdf_text([uploaded_file])
                st.subheader("Résultat de l'analyse")
                analysis_result = analyze_text_with_deepseek(raw_text, output_format)
                st.write(analysis_result)

    elif file_type == "Image":
        uploaded_file = st.file_uploader("Téléchargez votre CV (Image)", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            with st.spinner("Analyse de l'image..."):
                st.subheader("Résultat de l'analyse")
                analysis_result = analyze_image_with_gemini(image, output_format)
                st.write(analysis_result)

if __name__ == "__main__":
    main()