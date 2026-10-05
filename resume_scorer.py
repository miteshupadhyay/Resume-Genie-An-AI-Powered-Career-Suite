# app.py
import streamlit as st
from langchain_xai import ChatXAI
from langchain_community.document_loaders import PyPDFLoader
import os
import io
import tempfile
from langchain_groq import ChatGroq
from dotenv import load_dotenv


# ───────────────────────────────────────────────
#   CONFIG
# ───────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# You can also use st.secrets["XAI_API_KEY"] in production
XAI_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please set it in environment variables or st.secrets.")
    st.stop()

# The template we refined earlier
PROMPT_TEMPLATE = """You are an expert resume scorer and ATS optimization specialist with deep knowledge of recruitment practices across industries.

Task: Carefully analyze how well the candidate's resume matches the job description below. Base EVERY statement, score, and suggestion **strictly and exclusively** on the content actually present in the provided resume and job description. Do NOT invent, assume, or add any experience, skills, tools, achievements, or facts that are not explicitly written in the resume.

Job Description:
{job_description}

Candidate's Resume:
{context}

Produce the analysis using **exactly** the following structure and headings (do not add/remove sections, do not change headings):

Score: [integer]/100  
Overall Match: [integer]%  

Keywords matched:  
• [bullet list of important keywords/phrases from JD that DO appear in the resume]  

Missing keywords:  
• [bullet list of important/hard-required keywords/phrases from JD that are completely absent or extremely weakly represented in the resume]  

Readability Score: [integer]/100  
ATS Compatibility Score: [integer]/100  

2-liner summary:  
[One strong, concise sentence summarizing the overall fit]  
[One strong, concise sentence naming the single biggest current weakness]

Skill gap analysis:  
• [Bullet points – clear skill/tool/experience gaps, phrased as "Missing / weak: X → needed for Y part of the role"]  
• Focus on the most impactful gaps only (4–8 bullets max)

Overall improvement suggestions:  
• [Prioritized, actionable bullet points – strongest bang-for-buck improvements first]  
• Include both content (what to add/strengthen) and formatting/ATS tips

Industry specific feedback:  
• [2–5 bullets with observations tailored to this role’s industry / function – e.g. emphasis on certifications, specific metrics, project types, modern tools, regulatory knowledge, etc. Only include points that are genuinely relevant to the JD]

Scoring rubrics to follow (use your judgment applying these):
• Score (0–100)           → weighted combination of keyword presence, skill relevance, experience recency & level, achievements quantification, role progression
• Overall Match %         → rough estimated chance of passing initial ATS + recruiter screen
• Readability             → clarity, grammar, formatting, length, action verbs, density of fluff
• ATS Compatibility       → presence of standard section headings, keyword density (not stuffing), no tables/graphics, machine-readable layout cues

Be honest, direct, and constructive. If the match is very poor, say so clearly.
"""

# ───────────────────────────────────────────────
#   STREAMLIT APP
# ───────────────────────────────────────────────

st.set_page_config(page_title="Resume Scorer", layout="wide")

st.title("📄 Resume Matcher & Scorer")
st.markdown("Upload your resume (PDF) and paste the job description to get a detailed match analysis powered by Grok-4.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Job Description")
    job_description = st.text_area(
        "Paste the full job description here",
        height=320,
        placeholder="Responsibilities...\nRequirements...\nSkills...\n",
        key="jd_input"
    )

with col2:
    st.subheader("Your Resume")
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"], key="resume_uploader")

    if uploaded_file is not None:
        st.success("Resume uploaded ✓")

# ── Analyze button ────────────────────────────────────────

if st.button("Analyze Resume Match", type="primary", disabled=not (uploaded_file and job_description.strip())):

    with st.spinner("Extracting resume text..."):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            context = "\n\n".join(doc.page_content for doc in documents)

            os.unlink(tmp_path)  # clean up

        except Exception as e:
            st.error(f"Could not read the PDF: {e}")
            st.stop()

    if not context.strip():
        st.error("No readable text found in the resume PDF.")
        st.stop()

    # Build final prompt
    prompt = PROMPT_TEMPLATE.format(
        job_description=job_description.strip(),
        context=context.strip()
    )

    with st.spinner("Analyzing with Grok-4 (this can take 20–60 seconds)..."):

        try:
            chat = ChatGroq(
                model="qwen/qwen3.8-27b",
                temperature=0,
                api_key=os.getenv("GROQ_API_KEY")
            )

            response = chat.invoke(prompt)
            analysis_text = response.content

            st.subheader("📊 Resume Analysis Result")
            st.markdown(analysis_text)

        except Exception as e:
            st.error(f"API error: {str(e)}")
            if "rate limit" in str(e).lower():
                st.warning("Rate limit reached — please wait a few minutes and try again.")
            elif "authentication" in str(e).lower():
                st.error("Invalid or missing XAI_API_KEY.")