import os
import shutil
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


load_dotenv()

DB_dir = "career_coach_chroma_db"

def get_llm(model:str = "openai/gpt-oss-20b", temparature: float=0):
    try:
        api_key = os.getenv("GROQ_API_KEY")
    except:
        print("Model not loaded. Please try again.")


    return ChatGroq(model= model,
                    groq_api_key = api_key, 
                    temperature= temparature)

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def load_text_file(file_path: str, source_name: str, doc_type: str) ->List[Document]:

    path = path(file_path)
    text = path.read_text(encoding = "utf-8", errors="ignore")

    return [Document(page_content= text, metadata = {'source' : source_name, "doc_type" : doc_type})]


def create_documents(resume_text: str, jd_text: str) ->List[Document]:

    return [Document(page_content= resume_text,
                     metadata = {'source' : "uploaded_resume", "doc_type" : "resume"}),
            Document(page_content= jd_text, 
                     metadata = {'source' : "uploaded_jd", "doc_type" : "JD"})
                     ]

def split_documents(docs: List[Document], chunk_size = 1000, chunk_overlap: int = 150) ->List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size,
                                   chunk_overlap = chunk_overlap,
                                   separators= [ "\n\n","\n",".",""])
    return splitter.split_documents(docs)


def build_vectorstore(chunks: List[Document], persist_directory: str = DB_dir):
    if Path(persist_directory).exists():
        shutil.rmtree(persist_directory)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name="careerrag",
        persist_directory="career_coach_chroma_db"
    )
    return vectorstore


def retrieve_context(vectorstore, query:str, k: int = 3):
    retrieve = vectorstore.as_retriever(search_kwargs = {'k' : k})
    docs = retrieve.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])

    source_docs = docs

    return context, source_docs


def run_career_coach(vectorstore, resume_text: str, jd_text: str, question: str):
    llm = get_llm()
    retrieval_query = f"""Resume content and JD content relavent tot he career coaching question"""
    context, source_docs = retrieve_context(vectorstore,retrieval_query, k =3)

    prompt = ChatPromptTemplate.from_template("""
You are an expert AI Career Coach for students, freshers and working professionals.
Use Only the given context from the resume and job description.
Do not invent skills, experience or job requirements.

CONTEXT:
{context}

USER QUESTION:
{question}

Give a clear, practical answer with these sections when relevant:
1. Current Match Summary
2. Strengths
3. Missing Skills / Gaps
4. Recommanded Improvements
5. Suggested Projects
6. Interview Preparation Tips

keep the anser simple, actionable and beginner-friendly.
""")

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context" : context,
        "question" : question
    })

    return answer, source_docs

# To generate the complete Report.
def generate_complete_report(vectorstore, resume_text: str, jd_text: str):
    question = """
    Analyze this resume against this job description. Provide ATS-style, skill match,
    resume improvement suggestions, project suggestions, and interview questions.
    """
    return run_career_coach(vectorstore, resume_text, jd_text, question)