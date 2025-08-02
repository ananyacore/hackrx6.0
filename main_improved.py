#!/usr/bin/env python3
"""
HackRx 6.0 - LLM Document Processing System (Improved Version)
Enhanced with better prompt engineering and error handling
"""

import os
import uuid
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import PyPDF2
from docx import Document
import numpy as np
from llama_cpp import Llama

# Initialize LLM
print("Loading Llama model...")
try:
    llm = Llama(
        model_path="./llama-2-7b-chat.Q4_K_M.gguf",
        n_ctx=2048,
        n_threads=4,
        verbose=False
    )
    print("Llama model loaded successfully!")
except Exception as e:
    print(f"Error loading Llama model: {e}")
    llm = None

app = FastAPI(
    title="HackRx 6.0 - LLM Document Processing System (Improved)",
    description="Process policy documents and answer questions using local LLM with enhanced accuracy",
    version="2.0.0"
)

# In-memory storage
documents = {}  # doc_id -> {chunks, metadata}

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

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM response with multiple fallback strategies"""
    
    # Strategy 1: Find JSON block
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: Parse structured text response
    decision_match = re.search(r'(?:decision|Decision):\s*([a-zA-Z]+)', text, re.IGNORECASE)
    amount_match = re.search(r'(?:amount|Amount):\s*(\d+(?:\.\d+)?)', text)
    justification_match = re.search(r'(?:justification|Justification):\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    
    decision = decision_match.group(1).lower() if decision_match else "pending"
    amount = float(amount_match.group(1)) if amount_match else None
    justification = justification_match.group(1).strip() if justification_match else "Analysis completed"
    
    # Ensure decision is valid
    valid_decisions = ["approved", "rejected", "partial", "pending"]
    if decision not in valid_decisions:
        decision = "pending"
    
    return {
        "decision": decision,
        "amount": amount,
        "justification": justification,
        "clauses_used": [{"clause_text": "Policy analysis completed", "relevance_score": 0.8}],
        "confidence_score": 0.7
    }

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
        
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Store document
        documents[doc_id] = {
            'chunks': chunks,
            'full_text': text,
            'metadata': {
                'filename': file.filename,
                'upload_time': datetime.now().isoformat(),
                'num_chunks': len(chunks)
            }
        }
        
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
        
        if llm is None:
            raise HTTPException(status_code=500, detail="LLM model not loaded")
        
        doc_data = documents[request.doc_id]
        
        # Use relevant chunks (first 3 for better context)
        relevant_chunks = doc_data['chunks'][:3]
        context = "\n\n".join(relevant_chunks)
        
        # Enhanced prompt with better structure
        prompt = f"""You are an expert insurance policy analyzer. Analyze the policy document and answer the insurance claim question.

POLICY DOCUMENT:
{context}

CLAIM QUESTION: {request.question}

INSTRUCTIONS:
1. Determine if the claim should be APPROVED, REJECTED, PARTIAL, or PENDING
2. If approved/partial, estimate the coverage amount
3. Provide clear justification based on policy terms
4. Consider waiting periods, exclusions, and coverage limits

RESPONSE FORMAT (respond ONLY with this structure):
Decision: [approved/rejected/partial/pending]
Amount: [number or null]
Justification: [detailed explanation based on policy clauses]

Example:
Decision: approved
Amount: 50000
Justification: Knee surgery is covered under the policy with maximum benefit of $50,000. Patient meets the 30-day waiting period requirement.

YOUR ANALYSIS:"""

        # Generate response using Llama
        response = llm(
            prompt,
            max_tokens=300,
            temperature=0.1,
            stop=["\n\n", "USER:", "HUMAN:"]
        )
        
        # Extract response text
        response_text = response['choices'][0]['text'].strip()
        print(f"LLM Response: {response_text}")  # Debug logging
        
        # Parse the response
        parsed_response = extract_json_from_text(response_text)
        
        return AskResponse(
            decision=parsed_response.get("decision", "pending"),
            amount=parsed_response.get("amount"),
            justification=parsed_response.get("justification", "Analysis completed"),
            clauses_used=parsed_response.get("clauses_used", []),
            confidence_score=parsed_response.get("confidence_score", 0.7)
        )
        
    except Exception as e:
        print(f"Error in ask_question: {str(e)}")  # Debug logging
        # Return a fallback response instead of error
        return AskResponse(
            decision="pending",
            amount=None,
            justification=f"System error occurred during analysis: {str(e)}",
            clauses_used=[],
            confidence_score=0.0
        )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HackRx 6.0 - LLM Document Processing System (Improved)",
        "version": "2.0.0",
        "improvements": [
            "Enhanced prompt engineering",
            "Better JSON parsing",
            "Improved error handling",
            "Structured response format"
        ],
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
            "llm": "llama-2-7b-chat.Q4_K_M.gguf" if llm else "Not loaded"
        },
        "version": "2.0.0",
        "improvements": "Enhanced accuracy and error handling"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)