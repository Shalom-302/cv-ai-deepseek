# Analyseur de CV avec DeepSeek et Gemini

Ce projet utilise Streamlit, DeepSeek et Gemini pour analyser des CV au format PDF ou image. Il extrait des informations clés telles que l'éducation, l'expérience, les compétences, les langues, les certifications et les coordonnées.

## Fonctionnalités

* **Analyse de CV en PDF:** Extrait le texte des CV au format PDF et l'analyse à l'aide de DeepSeek.
* **Analyse de CV en image:**  Analyse les images de CV à l'aide de Gemini.
* **Extraction d'informations clés:**  Extrait les informations importantes du CV, notamment l'éducation, l'expérience, les compétences, les langues, les certifications et les coordonnées.
* **Interface utilisateur conviviale:**  Utilise Streamlit pour fournir une interface web interactive pour télécharger des CV et afficher les résultats de l'analyse.

## Installation

1. **Cloner le référentiel:**
   ```bash
   git clone https://github.com/Shalom-302/cv-ai-deepseek.git

Installer les dépendances:

pip install -r requirements.txt

Configurer les variables d'environnement:

Créer un fichier .env dans le répertoire racine du projet.
Ajouter les clés API DeepSeek et Google dans le fichier .env:

DEEPSEEK_API_KEY=votre_cle_deepseek
GOOGLE_API_KEY=votre_cle_google

Exécuter l'application Streamlit:

streamlit run main.py


Dépendances
streamlit
PyPDF2
langchain
openai
langchain_community
faiss-cpu
python-dotenv
Pillow
google-generativeai
Améliorations possibles
Améliorer la précision de l'extraction: Explorer d'autres modèles ou techniques d'extraction d'informations.
Ajouter la prise en charge de plusieurs langues: Prendre en charge l'analyse de CV dans des langues autres que l'anglais.
Intégrer une base de données: Stocker les résultats de l'analyse dans une base de données pour une recherche et une analyse ultérieures.
