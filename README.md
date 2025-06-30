```markdown
# 🧾 Automated Scheme Research Tool

This is a Streamlit-based application that lets users research Indian government schemes by uploading PDF/TXT files or pasting article URLs. It uses OpenAI's GPT-3.5 model, FAISS indexing, and LangChain to generate answers and summaries from the document content.

---

## 📌 Features

- Upload `.pdf` or `.txt` files **OR** paste URLs
- Extracts and processes scheme content using LangChain
- Generates contextual answers to user queries
- Provides an automatic bullet-point summary of each answer
- Displays source URLs
- Maintains a chat-style history of questions and answers

---

## 🖼 Demo

> 🎥 Demo video (replace with your GitHub or file link):

```

assets/demo.mp4

````

Embed in GitHub if uploading:
```markdown
![Demo](assets/demo.gif)
````

---

## 🚀 How to Run

### 1. Clone the repo

```bash
git clone https://github.com/your-username/haq_assignment.git
cd haq_assignment
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your OpenAI API key

Create a `.config` file in the root:

```ini
[OPENAI]
api_key = sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 4. Start the Streamlit app

```bash
streamlit run main.py
```

---

## 🧠 Tech Stack

* Python 3.10+
* Streamlit
* LangChain
* FAISS
* OpenAI GPT-3.5
* PyPDF2
* Requests

---

## 📂 Folder Structure

```
haq_assignment/
├── main.py                 # Streamlit interface
├── utils/
│   └── helpers.py          # Embeddings, query, loaders
├── .config                 # OpenAI API key config
├── requirements.txt
├── faiss_store_openai/     # FAISS vector store (auto-generated)
├── assets/
│   └── demo.mp4            # Screen recording or GIF
└── README.md
```

---

## ⚙️ .gitignore

Create this in your root to prevent versioning unnecessary files:

```
# Bytecode
__pycache__/
*.py[cod]

# Environments
venv/
.env

# PDFs and temp files
*.pdf
temp.pdf

# FAISS output
faiss_store_openai/

# Logs
*.log

# Editors
.vscode/
.idea/
.DS_Store
```

---

## ✅ Example Use

Ask:

* “Who is eligible for the Atal Pension Yojana?”
* “List the benefits of PMSVANidhi.”
* “What documents are required for application under this scheme?”

Upload or paste:

* [PMSVANidhi Guideline PDF](https://mohua.gov.in/upload/uploadfiles/files/PMSVANidhi%20Guideline_English.pdf)
* [NHB Schemes](https://nhb.gov.in/schemes.aspx)

---

## 🙋 Author

Developed by \[Your Name or GitHub Username]
Part of the **Haq Assignment Submission**

---

## 📄 License

For academic/demonstration use only.

```

---

