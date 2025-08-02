# HackRx 6.0 - LLM Document Processing System

A minimal FastAPI application that uses local LLM (Llama-2) to process policy documents and answer questions.

## Features

- **Document Upload**: Upload PDF, DOCX, or TXT files
- **Semantic Search**: Find relevant document chunks using embeddings
- **Local LLM**: Uses Llama-2 7B Chat model for question answering
- **Structured Responses**: Returns decisions, amounts, and justifications with clause references

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Llama model:**
   ```bash
   python download_llama.py
   ```

3. **Run the server:**
   ```bash
   python main.py
   ```

## API Endpoints

### POST /upload
Upload a document (PDF, DOCX, TXT)

**Response:**
```json
{
  "doc_id": "uuid-string",
  "message": "Document processed successfully. X chunks created.",
  "filename": "document.pdf"
}
```

### POST /ask
Ask a question about a specific document

**Request:**
```json
{
  "doc_id": "uuid-string",
  "question": "46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
  "chat_history": []
}
```

**Response:**
```json
{
  "decision": "approved",
  "amount": 50000,
  "justification": "Knee surgery is covered under policy clause 3.2.1",
  "clauses_used": [
    {
      "clause_text": "Knee surgery is covered under the policy.",
      "relevance_score": 0.92
    }
  ],
  "confidence_score": 0.85
}
```

## Usage Example

1. Upload a policy document:
   ```bash
   curl -X POST "http://localhost:8000/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@insurance_policy.pdf"
   ```

2. Ask a question:
   ```bash
   curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "doc_id": "your-doc-id",
       "question": "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
     }'
   ```

## API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

## Health Check

Visit `http://localhost:8000/health` to check system status. 