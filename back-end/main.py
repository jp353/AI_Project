from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
import fitz  # PyMuPDF
import docx

# Make sure to set your OpenAI API key as an environment variable or directly here
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Option 1: uses environment variable

# Uncomment below to hardcode your key (Option 2 - not recommended for production)
# client = OpenAI(api_key="sk-...")

app = FastAPI()

# Allow frontend access during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to specific origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text(file: UploadFile) -> str:
    if file.filename.endswith(".pdf"):
        pdf = fitz.open(stream=file.file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in pdf])
    elif file.filename.endswith(".docx"):
        doc = docx.Document(file.file)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        return ""

@app.post("/summarize")
async def summarize(
    file: UploadFile = File(...),
    prompt: str = Form("Summarize this document")
):
    text = extract_text(file)
    if not text:
        return {"summary": "Unable to extract text."}

    # Truncate to avoid token limits (~3000 chars max for gpt-3.5-turbo)
    text = text[:3000]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}\n\n{text}"}
        ]
    )

    return {"summary": response.choices[0].message.content.strip()}
