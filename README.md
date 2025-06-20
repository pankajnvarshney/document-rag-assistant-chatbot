# RAG Chatbot Project

SUMMARY 

This project demonstrates the development of an AI-powered chatbot using a Retrieval Augmented Generation (RAG) architecture to answer questions based on a long-form document (AI_Training_Document.pdf). It combines a semantic retriever using FAISS and MiniLM embeddings with a lightweight language model (flan-t5-base) to generate grounded, context-aware responses. The application features a real-time chat interface built using 
Streamlit. 

---
PROJECT STRUCTURE 

rag_chatbot/

├── app.py                         # Streamlit chatbot interface


├── requirements.txt               # Python dependencies


├── README.md                      # Project documentation
│

├── /data/                         # Original document

│   └── AI_Training_Document.pdf
│

├── /chunks/                       # Chunked text data (pickle format)

│   └── chunks.pkl
│

├── /vectordb/                     # FAISS vector index

│   ├── index.faiss

│   └── chunks.pkl
│

├── /notebooks/                    # Preprocessing notebook

│   └── preprocessing.ipynb
│

├── /src/                          # Core RAG pipeline

│   ├── retriever.py               # Loads FAISS & retrieves top-k chunks

│   ├── generator.py               # Loads model & generates answers

│   └── rag_pipeline.py            # Combined retriever + generator logic

---

How to Run Locally

### 1. Clone the Repository

```
git clone https://github.com/your-username/rag-chatbot-doc-qa.git
cd rag-chatbot-doc-qa
```

### 2. Create a Virtual Environment

```
python -m venv venv
venv\Scripts\activate    # On Windows
```

### 3. Install Requirements

```
pip install -r requirements.txt
```

### 4. Preprocess the Document

Run the preprocessing notebook or script to:
- Clean & chunk the PDF
- Generate embeddings
- Create FAISS vector index

```
jupyter notebook notebooks/preprocessing.ipynb
# or
python src/preprocess.py
```

### 5. Launch the Chatbot

```
streamlit run app.py
```

---

## 💬 Example Queries

- "What are the rules for using eBay services?"
- "Who is eligible to create an eBay account?"
- "Can I sell a vehicle on eBay?"
- "What is the eBay Money Back Guarantee?"

---

## 📌 Notes

- This version uses `flan-t5-base` to keep memory usage low (CPU-friendly)
- Source chunks used in answers are optionally viewable in the UI
- Ideal for internal document search, policy QA bots, or compliance assistants

---



Use Streamlit to ask questions about a PDF using RAG.
