import streamlit as st
import io
import matplotlib.pyplot as plt
import mymodule

st.title("Resume Parser & Job Description Comparator")

# --- Resume Upload & Parsing ---
st.header("Upload Your Resume")
uploaded_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")

if uploaded_file is not None:
    # Extract text from the uploaded PDF file
    resume_text = mymodule.pdf_to_text(uploaded_file)
    
    st.subheader("Extracted Resume Text")
    st.text_area("Resume Text", value=resume_text, height=200)
    
    # Parse resume fields using your extraction functions
    parsed_info = {
        "Name": mymodule.extract_name_hybrid(resume_text),
        "Email": mymodule.extract_email_ner(resume_text),
        "Phone": mymodule.extract_phone_ner(resume_text),
        "Education": mymodule.extract_education(resume_text),
        "Skills": mymodule.extract_skills(resume_text)
    }
    
    st.subheader("Parsed Resume Information")
    st.markdown(f"**Name:** {parsed_info['Name'] or 'Not Found'}")
    st.markdown(f"**Email:** {parsed_info['Email'] or 'Not Found'}")
    st.markdown(f"**Phone:** {parsed_info['Phone'] or 'Not Found'}")
    st.markdown(f"**Education:** {', '.join(parsed_info['Education']) if parsed_info['Education'] else 'Not Found'}")
    st.markdown(f"**Skills:** {', '.join(parsed_info['Skills']) if parsed_info['Skills'] else 'Not Found'}")
    
    # --- Job Description Comparison ---
    st.header("Compare with Job Description")
    jd_text = st.text_area("Paste Job Description Here", height=200)
    
    if st.button("Compare"):
        # Compare the extracted skills with the job description
        similarity_score = mymodule.compare_with_jd(jd_text.lower(), parsed_info["Skills"])
        st.subheader("Comparison Result")
        st.markdown(f"**Similarity Score:** {round(similarity_score * 100, 2)}%")
        
        # Display a knowledge graph of matched skills
        st.subheader("Knowledge Graph")
        # Create an in-memory buffer to save the graph image
        buf = io.BytesIO()
        mymodule.draw_knowledge_graph(jd_text, parsed_info["Skills"])
        plt.savefig(buf, format="png")
        buf.seek(0)
        st.image(buf)
        plt.clf()  # Clear the current figure to avoid overlapping in subsequent runs
