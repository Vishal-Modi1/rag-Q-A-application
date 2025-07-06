```markdown
# 🧠 RAG Document Q&A App

A Streamlit-powered application that uses **LangChain**, **FAISS**, and **Groq's LLaMA model** to answer questions from your uploaded PDF documents. It follows a **Retrieval-Augmented Generation (RAG)** architecture and supports embeddings via **OpenAI**.

---

## 🚀 Features

- Paste your pdfs inside documents folder
- Uses **Groq’s LLaMA 3.1 8B Instant** for lightning-fast responses
- Embeds documents using **OpenAI Embeddings**
- Saves and reuses FAISS vector index to minimize cost
- Clean UI built with **Streamlit**

---

## 🛠️ Tech Stack

- **Streamlit** for UI
- **LangChain** for RAG pipeline
- **FAISS** for vector storage
- **OpenAI** for embeddings
- **Groq** (LLaMA) for LLM inference

---

## 📁 Folder Structure

```

project-root/
├── documents/              # Place your PDFs here
├── faiss\_index/            # Saved vector DB (auto-generated)
├── .env                    # API keys (pushed with blank values)
├── app.py                  # Main Streamlit app
├── rag-requirements.txt    # Python dependencies

````
## 🔐 Environment Variables

Create a `.env` file in the project root with the following keys and add actual values

```env
LANGCHAIN_API_KEY=""
GROQ_API_KEY=""
OPENAI_API_KEY=""
````

> ✅ These keys are **included in the repo** with **blank values** — no sensitive info is exposed.

---

## 🧪 Installation & Running the App

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/rag-doc-qa.git
cd rag-doc-qa
```

2. **Create a virtual environment and activate it:**

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r rag-requirements.txt
```

4. **Add your PDFs to the `documents/` folder.**

5. **Run the Streamlit app:**

```bash
streamlit run app.py
```

---

## 💡 How It Works

1. Loads PDFs from the `documents/` folder
2. Splits and embeds text using OpenAI
3. Stores embeddings in FAISS (saved to disk)
4. Retrieves relevant chunks via LangChain retriever
5. Sends user query + context to Groq LLaMA for answering

---

## 🧑‍💻 Author

**Vishal Modi**
Senior .NET Developer & Generative AI Enthusiast (Developer + Lifelong Learner)
[GitHub](https://github.com/your-username) • [LinkedIn]([https://linkedin.com/in/your-profile](https://www.linkedin.com/in/vishal-modi-7995a5106/))
