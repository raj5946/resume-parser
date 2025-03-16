import re
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import pdfplumber

# Load pre-trained spaCy model
nlp = spacy.load("en_core_web_sm")

# ===================================EXTRACTING TEXT FROM PDF======================================#
def pdf_to_text(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

# ===================================EXTRACTING NAME (NER+RE)======================================#
def extract_name_hybrid(text):
    name_pattern = r"^([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)"
    match = re.match(name_pattern, text)
    regex_name = match.group().strip() if match else None

    doc = nlp(text)
    ner_name = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), None)

    return ner_name if ner_name else regex_name

# ===================================EXTRACTING EMAIL (NER+RE)======================================#
def extract_email_ner(text):
    doc = nlp(text)
    ner_email = next((ent.text for ent in doc.ents if ent.label_ == "EMAIL"), None)
    if ner_email:
        return ner_email
    
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    match = re.search(email_pattern, text)
    if match:
        return match.group()
    
    indeed_pattern = r"indeed\.com/r/[\w-]+/\w+"
    indeed_match = re.search(indeed_pattern, text)
    if indeed_match:
        return indeed_match.group()
    
    return None

# ===================================EXTRACTING PHONE NO. (NER+RE)======================================#
def extract_phone_ner(text):
    doc = nlp(text)
    ner_phone = next((ent.text for ent in doc.ents if ent.label_ == "PHONE"), None)
    if ner_phone:
        return ner_phone
    
    phone_pattern = r"""
        (?:(?:\+?\d{1,3})?[\s.-]?)?  
        \(?\d{3}\)?[\s.-]?           
        \d{3}[\s.-]?\d{4}            
    """
    match = re.search(phone_pattern, text, re.VERBOSE)
    return match.group().strip() if match else None

# ===================================EXTRACTING EDUCATION (NER)======================================#
def extract_education(doc):
    universities = []
    doc = nlp(doc)
    for entity in doc.ents:
        if entity.label_ == "ORG" and ("university" in entity.text.lower() or "college" in entity.text.lower() or "institute" in entity.text.lower()):
            universities.append(entity.text)
    return universities

# ===================================EXTRACTING SKILLS (NER TRAINED MODEL)======================================#
def extract_skills(text):
    # Load the trained model
    nlp_skill = spacy.load("skill_ner_modelV1.2")
    doc = nlp_skill(text)
    # Extract skills
    extracted_skills = [ent.text.lower() for ent in doc.ents if ent.label_ == "SKILL"]
    return extracted_skills

# ===================================GENERATE SKILL CLOUD======================================#
def generate_skill_cloud(skills):
    skills_text = " ".join(skills)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(skills_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Skill Cloud", fontsize=14)
    plt.show()

# ===================================COMPARISON WITH JOB DESCRIPTION======================================#
def compare_with_jd(jd_text, extracted_skills):
    """Compares extracted skills with the job description and returns a similarity score."""
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([jd_text, " ".join(extracted_skills)])
    similarity_score = cosine_similarity(vectors)[0, 1]
    return similarity_score

# ===================================GENERATING KNOWLEDGE GRAPH======================================#
def draw_knowledge_graph(jd_text, extracted_skills):
    """Creates a knowledge graph with two main nodes: 'Resume' and 'Job Description', and nodes for skills."""
    G = nx.Graph()
    # For a simple JD skills extraction, we use our extract_skills function on the job description.
    jd_skills = extract_skills(jd_text.lower())
    G.add_node("Resume", size=1500, color='skyblue')
    G.add_node("Job Description", size=1500, color='orange')

    for skill in extracted_skills:
        G.add_node(skill, size=700, color='lightgreen')
        G.add_edge("Resume", skill, color='gray')

    for skill in jd_skills:
        G.add_node(skill, size=700, color='lightcoral')
        G.add_edge("Job Description", skill, color='gray')

    common_skills = set(extracted_skills).intersection(jd_skills)
    for skill in common_skills:
        G.add_edge("Resume", skill, color='blue')
        G.add_edge("Job Description", skill, color='blue')

    pos = nx.spring_layout(G, seed=42)
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    node_sizes = [G.nodes[n]['size'] for n in G.nodes()]
    node_colors = [G.nodes[n]['color'] for n in G.nodes()]

    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, with_labels=True, edge_color=edge_colors, node_color=node_colors, node_size=node_sizes, font_size=10)
    plt.title("Knowledge Graph: Resume vs Job Description Skills Matching")
    plt.show()
