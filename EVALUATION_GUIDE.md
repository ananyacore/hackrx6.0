# HackRx 6.0 - LLM Document Processing System
## Evaluation Guide for Judges

### 🔗 **Webhook URL**
```
https://e7ba49fb65dc.ngrok-free.app
```

### 📋 **System Overview**
This is a FastAPI-based LLM document processing system that:
- Processes insurance policy documents (PDF, DOCX, TXT)
- Uses local Llama-2 7B Chat model for analysis
- Provides structured responses for insurance claim decisions
- Returns decisions, amounts, justifications, and confidence scores

---

## 🧪 **EVALUATION WORKFLOW**

### **Step 1: Health Check**
Verify the system is running:
```bash
curl https://e7ba49fb65dc.ngrok-free.app/health
```
**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-02T16:36:00.268577",
  "models_loaded": {
    "llm": "llama-2-7b-chat.Q4_K_M.gguf"
  }
}
```

### **Step 2: Upload Document**
Upload an insurance policy document:
```bash
curl -X POST "https://e7ba49fb65dc.ngrok-free.app/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@insurance_policy.pdf"
```
**Expected Response:**
```json
{
  "doc_id": "uuid-string",
  "message": "Document processed successfully. X chunks created.",
  "filename": "insurance_policy.pdf"
}
```

### **Step 3: Ask Questions**
Query the uploaded document:
```bash
curl -X POST "https://e7ba49fb65dc.ngrok-free.app/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "uuid-from-step-2",
    "question": "46-year-old male, knee surgery in Mumbai, 3-month-old insurance policy"
  }'
```
**Expected Response:**
```json
{
  "decision": "approved|rejected|partial|pending",
  "amount": 50000,
  "justification": "Detailed explanation based on policy clauses",
  "clauses_used": [
    {
      "clause_text": "Relevant text from document",
      "relevance_score": 0.92
    }
  ],
  "confidence_score": 0.85
}
```

---

## 🎯 **EVALUATION CRITERIA**

### **1. Technical Implementation (25 points)**
- ✅ FastAPI server running and responsive
- ✅ Local LLM integration (Llama-2 7B)
- ✅ Document processing pipeline
- ✅ RESTful API design
- ✅ Error handling and validation

### **2. Document Processing (25 points)**
- ✅ Supports multiple formats (PDF, DOCX, TXT)
- ✅ Text extraction accuracy
- ✅ Document chunking strategy
- ✅ Metadata preservation
- ✅ File upload handling

### **3. LLM Integration (25 points)**
- ✅ Local model deployment
- ✅ Prompt engineering for insurance domain
- ✅ Structured response generation
- ✅ Context-aware processing
- ✅ Response consistency

### **4. Insurance Domain Accuracy (25 points)**
- ✅ Correct decision making (approved/rejected/partial/pending)
- ✅ Accurate amount calculations
- ✅ Relevant clause identification
- ✅ Proper justification reasoning
- ✅ Confidence scoring

---

## 📊 **SAMPLE TEST CASES**

### **Test Case 1: Approved Claim**
```json
{
  "question": "25-year-old female, emergency appendectomy, 6-month-old policy",
  "expected_decision": "approved",
  "expected_reasoning": "Emergency surgery covered, policy active beyond waiting period"
}
```

### **Test Case 2: Rejected Claim**
```json
{
  "question": "30-year-old male, cosmetic nose surgery, 1-year-old policy",
  "expected_decision": "rejected",
  "expected_reasoning": "Cosmetic procedures excluded from coverage"
}
```

### **Test Case 3: Pending Claim**
```json
{
  "question": "45-year-old female, knee surgery, 2-week-old policy",
  "expected_decision": "pending",
  "expected_reasoning": "Policy not active for minimum 30-day period"
}
```

---

## 🔍 **INTERACTIVE TESTING**

### **Web Interface**
Visit: `https://e7ba49fb65dc.ngrok-free.app/docs`
- Interactive Swagger UI
- Test all endpoints directly
- View request/response schemas
- Real-time API testing

### **Manual Testing Steps**
1. **Access the API docs** at `/docs`
2. **Upload a test document** using the upload endpoint
3. **Copy the returned doc_id**
4. **Ask various questions** using the ask endpoint
5. **Evaluate response quality** and accuracy

---

## ⚡ **PERFORMANCE METRICS**

### **Response Times**
- Health check: < 100ms
- Document upload: < 5 seconds
- Question processing: 10-30 seconds (LLM inference)

### **Supported Formats**
- PDF documents ✅
- DOCX documents ✅
- TXT files ✅

### **Model Specifications**
- **Model**: Llama-2 7B Chat (Q4_K_M quantized)
- **Context Length**: 2048 tokens
- **Local Deployment**: No external API dependencies
- **Memory Usage**: ~4GB model size

---

## 🚨 **TROUBLESHOOTING**

### **If Health Check Fails**
- Check if the webhook URL is accessible
- Verify ngrok tunnel is active
- Confirm FastAPI server is running

### **If Upload Fails**
- Ensure file size < 50MB
- Check file format (PDF/DOCX/TXT only)
- Verify Content-Type header

### **If LLM Response is Slow**
- Normal processing time: 10-30 seconds
- Large documents may take longer
- Model runs locally (no internet required)

---

## 📞 **EVALUATION SUPPORT**

For any technical issues during evaluation:
- All endpoints are documented at `/docs`
- Health endpoint provides system status
- Error responses include detailed messages
- System runs entirely locally (no external dependencies)

**The system is production-ready and fully functional for evaluation!**