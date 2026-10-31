# 🤖 Multi-Agent AI Customer Support System

A functional, multi-agent customer support platform built with **LangChain**, **Groq**, **Sentence-Transformers**, **FAISS**, and **Streamlit**. The system leverages a neural network classifier on Banking77 intents to route customer inquiries across four dedicated domain agents with real-time observability.

---

## 🏗 System Architecture & Data Flow

```text
User Query (Streamlit UI)
       │
       ▼
┌──────────────┐
│  LSTM Router │ ──► Predicts Intent (Billing, Technical, Booking, General)
└──────┬───────┘
       │
       ├─────────────────┬──────────────────┬─────────────────┐
       ▼                 ▼                  ▼                 ▼
┌───────────────┐   ┌─────────────┐    ┌─────────────┐   ┌─────────────┐
│ Billing Agent │   │Booking Agent│    │Technical Agt│   │ General Agt │
│  (Tool Call)  │   │ (Issuance)  │    │ (RAG Vector)│   │ (RAG Vector)│
└───────────────┘   └─────────────┘    └─────────────┘   └─────────────┘
                                              │                 │
                                              ▼                 ▼
                                       ┌──────────────┐  ┌──────────────┐
                                       │ FAISS Index  │  │ FAISS Index  │
                                       │ (tech docs)  │  │  (faq docs)  │
                                       └──────────────┘  └──────────────┘
```
---

## How Data Moves:
1. **User Interaction:** The user submits a prompt via the Streamlit web frontend (`app.py`).
2. **Intent Classification:** The query passes to `router.py`, where the neural router processes the input text and predicts the query intent along with a confidence score.
3. **Dynamic Routing:**
   * **`billing`** & **`booking`**: Routed to specialized agents equipped with action tools (e.g., account lookups, card orders/deliveries).
   * **`technical`** & **`general`**: Routed to RAG-enabled agents querying FAISS vector stores containing chunked knowledge bases (`docs/technical/` and `docs/general/`).
4. **LLM Inference:** Agents format tool outputs or retrieved document contexts into prompts sent to Groq (`openai/gpt-oss-20b`).
5. **Observability Render:** Streamlit displays the agent's answer alongside real-time metrics (Predicted Intent, Confidence %, Active Agent) in the sidebar.

# 📁 Directory Structure for Gen AI Project

```text
📁 Directory Gen AI/
├── agents/
│   ├── billing.py            # Billing domain agent & tool schemas
│   ├── booking.py            # Booking/Card issuance agent
│   ├── technical.py          # Technical support RAG agent
│   ├── general.py            # General FAQ RAG agent
│   └── rag_utils.py          # Embedding loader & FAISS retriever setup
├── dataset/
│   └── categories.json       # Banking77 intent category index mappings
│   ├── test.csv              # Test Dataset
│   └── train.csv             # Train Dataset
├── docs/
│   ├── technical/
│   │   └── troubleshooting.txt
│   └── general/
│       └── faq.txt
├── .env                      # API keys (GROQ_API_KEY)
├── app.py                    # Streamlit frontend & sidebar observability
├── router.py                 # Multi-agent routing logic & classifier wrapper
└── requirements.txt          # Dependencies list
```

## ⚙️ Setup & Installation

### 1. Prerequisites

- Python 3.10 – 3.12 recommended.
- A valid Groq API Key.

### 2. Environment Setup

Clone the repository and create a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```
Install dependencies:
```powershell
pip install langchain langchain-community sentence-transformers faiss-cpu streamlit python-dotenv langchain-groq torch
```

```powershell
pip install langchain langchain-community sentence-transformers faiss-cpu streamlit python-dotenv langchain-groq torch
```

### 3. Configure API Key
Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## 🚀 How to Run the Application
### Launch Streamlit Frontend
```powershell
streamlit run app.py
```
The application will open automatically in your browser at http://localhost:8501.

## 🧪 Example Test Queries

| Domain    | Intent                            | Example Query                                                                         |
|-----------|-----------------------------------|---------------------------------------------------------------------------------------|
| Billing   | Extra Charges / Refunds           | "Why was I charged a $15 foreign transaction fee on my statement?"                    |
| Billing   | Duplicate Charge / Exchange Rates | "I was charged twice for my subscription payment yesterday, can I get a refund?"      |
| Booking   | Card Issuance & Orders            | "How do I order a new physical card and get a delivery estimate?"                     |
| Booking   | Virtual Cards / Card Expiry       | "My card is about to expire, how do I generate a new virtual disposable card?"        |
| Technical | Troubleshooting (RAG)             | "How do I unblock my PIN after entering it wrong 3 times at an ATM?"                  |
| Technical | Troubleshooting (RAG)             | "My app keeps crashing whenever I try to link my credit card, what should I do?"      |
| General   | FAQ & Policies (RAG)              | "What is the minimum age requirement to open an account?"                             |
| General   | FAQ & Policies (RAG)              | "Which countries and currencies are currently supported for international transfers?" |
