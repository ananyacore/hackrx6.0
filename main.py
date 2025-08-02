#!/usr/bin/env python3
"""
HackRx 6.0 - LLM Document Processing System
Minimal FastAPI application for processing policy documents with local Llama model
"""

import os
import uuid
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import PyPDF2
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from llama_cpp import Llama

# Initialize models
print("Loading models...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
llm = Llama(
    model_path="./llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=4,
    verbose=False
)
print("Models loaded successfully!")

app = FastAPI(
    title="HackRx 6.0 - LLM Document Processing System",
    description="Process policy documents and answer questions using local LLM",
    version="1.0.0"
)

# In-memory storage
documents = {}  # doc_id -> {chunks, embeddings, metadata}
faiss_index = None
chunk_texts = []

class AskRequest(BaseModel):
    doc_id: str
    question: str
    chat_history: Optional[List[Dict[str, str]]] = None

class AskResponse(BaseModel):
    decision: str
    amount: Optional[float] = None
    justification: str
    clauses_used: List[Dict[str, Any]]
    confidence_score: float

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file"""
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    words = text.split()
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

def create_faiss_index(embeddings: List[np.ndarray]) -> faiss.IndexFlatIP:
    """Create FAISS index for similarity search"""
    embeddings_array = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatIP(embeddings_array.shape[1])
    index.add(embeddings_array)
    return index

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.pdf', '.docx', '.txt')):
            raise HTTPException(status_code=400, detail="Only PDF, DOCX, and TXT files are supported")
        
        # Save file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Extract text based on file type
        if file.filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(temp_path)
        elif file.filename.lower().endswith('.docx'):
            text = extract_text_from_docx(temp_path)
        else:  # .txt
            with open(temp_path, 'r', encoding='utf-8') as f:
                text = f.read()
        
        # Clean up temp file
        os.remove(temp_path)
        
        # Chunk the text
        chunks = chunk_text(text)
        
        # Create embeddings
        embeddings = embedding_model.encode(chunks)
        
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Store document
        documents[doc_id] = {
            'chunks': chunks,
            'embeddings': embeddings,
            'metadata': {
                'filename': file.filename,
                'upload_time': datetime.now().isoformat(),
                'num_chunks': len(chunks)
            }
        }
        
        # Update FAISS index
        global faiss_index, chunk_texts
        all_embeddings = []
        chunk_texts = []
        
        for doc_data in documents.values():
            all_embeddings.extend(doc_data['embeddings'])
            chunk_texts.extend(doc_data['chunks'])
        
        if all_embeddings:
            faiss_index = create_faiss_index(all_embeddings)
        
        return {
            "doc_id": doc_id,
            "message": f"Document processed successfully. {len(chunks)} chunks created.",
            "filename": file.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Ask a question about a specific document"""
    try:
        # Check if document exists
        if request.doc_id not in documents:
            raise HTTPException(status_code=404, detail="Document not found")
        
        doc_data = documents[request.doc_id]
        
        # Create query embedding
        query_embedding = embedding_model.encode([request.question])
        
        # Search for relevant chunks
        if faiss_index is not None:
            D, I = faiss_index.search(query_embedding, min(5, len(chunk_texts)))
            relevant_chunks = [chunk_texts[i] for i in I[0]]
        else:
            # Fallback: use all chunks from the document
            relevant_chunks = doc_data['chunks'][:5]
        
        # Prepare context for LLM
        context = "\n\n".join(relevant_chunks)
        
        # Create prompt for Llama
        system_prompt = """You are an expert insurance policy analyzer. Your task is to analyze the provided document context and answer questions about insurance policies. 

You must provide a structured response with:
1. Decision: "approved", "rejected", "partial", or "pending"
2. Amount: numeric value if applicable, otherwise null
3. Justification: clear explanation of your decision
4. Confidence: score between 0 and 1

Focus on the specific clauses and rules mentioned in the document context."""

        user_prompt = f"""Document Context:
{context}

Question: {request.question}

Please provide a JSON response with the following structure:
{{
    "decision": "approved|rejected|partial|pending",
    "amount": <numeric_value_or_null>,
    "justification": "detailed explanation",
    "clauses_used": [
        {{
            "clause_text": "relevant text from document",
            "relevance_score": <float_between_0_and_1>
        }}
    ],
    "confidence_score": <float_between_0_and_1>
}}"""

        # Generate response using Llama
        response = llm(
            f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]",
            max_tokens=512,
            temperature=0.1,
            stop=["</s>", "[INST]"]
        )
        
        # Extract response text
        response_text = response['choices'][0]['text'].strip()
        
        # Try to parse JSON from response
        try:
            # Find JSON in response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                parsed_response = json.loads(json_str)
            else:
                # Fallback response
                parsed_response = {
                    "decision": "pending",
                    "amount": None,
                    "justification": "Unable to parse response from LLM",
                    "clauses_used": [],
                    "confidence_score": 0.0
                }
        except json.JSONDecodeError:
            # Fallback response
            parsed_response = {
                "decision": "pending",
                "amount": None,
                "justification": f"LLM Response: {response_text}",
                "clauses_used": [],
                "confidence_score": 0.0
            }
        
        return AskResponse(
            decision=parsed_response.get("decision", "pending"),
            amount=parsed_response.get("amount"),
            justification=parsed_response.get("justification", "No justification provided"),
            clauses_used=parsed_response.get("clauses_used", []),
            confidence_score=parsed_response.get("confidence_score", 0.0)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HackRx 6.0 - LLM Document Processing System",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /upload - Upload a document",
            "ask": "POST /ask - Ask a question about a document"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "embedding": "all-MiniLM-L6-v2",
            "llm": "llama-2-7b-chat.Q4_K_M.gguf"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 