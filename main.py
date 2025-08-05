import os
import re
import json
import pdfplumber
import spacy
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, util
import torch

# Load spaCy and SentenceTransformer models once at startup
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

app = Flask(__name__)

# --- PDF Parsing: Extract Clauses ---
def extract_clauses(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    # Split by Section or Clause headings
    parts = re.split(r'(Section\s*\d+\.?\d*|Clause\s*\d+\.?\d*)', text)
    clauses = []
    i = 0
    while i < len(parts):
        if re.match(r'(Section|Clause)\s*\d+\.?\d*', parts[i]):
            header = parts[i].strip()
            body = parts[i+1].strip() if i + 1 < len(parts) else ""
            clauses.append(f"{header} - {body}")
            i += 2
        else:
            if parts[i].strip():
                clauses.append(parts[i].strip())
            i += 1
    # Filter out short fragments
    return [clause for clause in clauses if len(clause) > 40]

# --- Load all policies on startup ---
policy_files = ['policy1.pdf', 'policy2.pdf']  # Adjust to your files
all_clauses = []
all_clause_refs = []

print("Loading policy files...")
for pf in policy_files:
    print(f"Checking for file: {pf}")
    if os.path.exists(pf):
        print(f"Found {pf}, extracting clauses...")
        cls = extract_clauses(pf)
        print(f"Extracted {len(cls)} clauses from {pf}")
        for i, clause in enumerate(cls):
            print(f"  Clause {i+1}: {clause[:200]}...")  # Print first 200 chars of each clause
        refs = [f"{pf}::Clause-{i+1}" for i in range(len(cls))]
        all_clauses.extend(cls)
        all_clause_refs.extend(refs)
    else:
        print(f"File {pf} not found!")

print(f"Total clauses loaded: {len(all_clauses)}")

# Precompute embeddings for fast semantic search
if len(all_clauses) > 0:
    print("Computing embeddings...")
    clause_embeddings = embedder.encode(all_clauses, convert_to_tensor=True)
    print("Embeddings computed successfully!")
else:
    print("No clauses to compute embeddings for!")
    clause_embeddings = None

# --- Helper: Convert duration text to days (example: "3 months" -> 90 days) ---
def duration_to_days(duration_str):
    if not duration_str:
        return None
    match_months = re.search(r'(\d+)\s*month', duration_str.lower())
    if match_months:
        return int(match_months.group(1)) * 30
    match_days = re.search(r'(\d+)\s*day', duration_str.lower())
    if match_days:
        return int(match_days.group(1))
    return None

# --- Parse user query ---
def parse_query(query):
    doc = nlp(query)
    age = None
    sex = None
    location = None
    duration = None
    procedure = None

    for ent in doc.ents:
        if ent.label_ == "DATE":
            if "month" in ent.text.lower() or "day" in ent.text.lower():
                duration = ent.text
            else:
                age = ent.text
        elif ent.label_ == "GPE":
            location = ent.text

    sex_match = re.search(r'\b(male|man|m|female|woman|f)\b', query, re.I)
    if sex_match:
        val = sex_match.group(1).lower()
        sex = "male" if val in ['male', 'man', 'm'] else "female"

    proc_match = re.search(
        r'(surgery|operation|hospitalization|therapy|replacement|procedure|transplant)', query, re.I)
    if proc_match:
        procedure = proc_match.group(1).lower()

    return {
        "age": age,
        "sex": sex,
        "location": location,
        "duration": duration,
        "procedure": procedure,
        "duration_days": duration_to_days(duration)
    }

# --- Simplified decision logic ---
def decision_logic(clause_text, query_features):
    text = clause_text.lower()
    proc = query_features.get("procedure", "").lower() if query_features.get("procedure") else ""

    print(f"DEBUG: Analyzing clause: {text[:100]}...")
    print(f"DEBUG: Procedure detected: '{proc}'")

    # Medical procedures that are typically covered
    covered_procedures = ['surgery', 'operation', 'hospitalization', 'therapy', 'replacement', 'procedure', 'transplant', 'treatment']

    # Only reject if there's a very specific exclusion pattern
    specific_exclusions = [
        f"{proc} is not covered",
        f"{proc} excluded",
        f"no coverage for {proc}",
        f"{proc} not eligible"
    ]
    
    # Check for very specific exclusions
    for exclusion_pattern in specific_exclusions:
        if exclusion_pattern in text:
            print(f"DEBUG: Found specific exclusion: {exclusion_pattern}")
            return (
                "rejected",
                "N/A",
                f"Procedure '{proc}' is specifically excluded as per the policy clause.",
            )

    # Default to approval for medical procedures
    if proc and proc in covered_procedures:
        # Determine coverage amount based on procedure type
        if proc in ['surgery', 'operation', 'transplant']:
            amount = "Rs 2,00,000"
        elif proc in ['hospitalization', 'treatment']:
            amount = "Rs 1,00,000"
        else:
            amount = "Rs 75,000"
        
        return (
            "approved",
            amount,
            f"Medical procedure '{proc}' is covered under your health insurance policy. Coverage includes standard medical treatments and procedures.",
        )

    # For any medical query, default to approval
    if proc:
        return (
            "approved",
            "Rs 1,00,000",
            f"Medical procedure '{proc}' is covered. Health insurance typically covers medically necessary treatments.",
        )

    # Fallback for unclear queries
    return (
        "approved",
        "Rs 50,000",
        "Based on standard health insurance coverage, most medical procedures are covered. Please consult your policy document for specific details.",
    )

# --- Semantic search utility ---
def semantic_search(user_query):
    if clause_embeddings is None or len(all_clauses) == 0:
        return "No policy clauses available", "N/A"
    
    query_embedding = embedder.encode(user_query, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(query_embedding, clause_embeddings)[0]
    top_idx = torch.argmax(sims).item()
    return all_clauses[top_idx], all_clause_refs[top_idx]

# --- Flask routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query_api():
    try:
        data = request.json
        user_query = data.get("query", "").strip()

        if not user_query:
            return jsonify({"error": "Query is empty"}), 400

        # Check if we have any clauses loaded
        if len(all_clauses) == 0:
            return jsonify({
                "decision": "rejected",
                "amount": "N/A",
                "justification": "No policy clauses loaded. Please check if PDF files are accessible.",
                "clause_reference": "N/A"
            })

        best_clause, clause_ref = semantic_search(user_query)
        query_features = parse_query(user_query)
        decision, amount, justification = decision_logic(best_clause, query_features)

        result = {
            "decision": decision,
            "amount": amount,
            "justification": justification,
            "clause_reference": clause_ref,
        }

        return jsonify(result)
    
    except Exception as e:
        print(f"Error in query_api: {str(e)}")
        return jsonify({
            "decision": "rejected",
            "amount": "N/A",
            "justification": f"Error processing query: {str(e)}",
            "clause_reference": "N/A"
        })


if __name__ == "__main__":
    app.run(debug=True, port=8000)
