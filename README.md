# 📄 Chat with Your PDF - RAG Application

A powerful PDF chat application built with **LangChain**, **Groq**, and **Streamlit**. Upload any PDF and ask questions about its content using Retrieval-Augmented Generation (RAG).

![Python](https://img.shields.io/badge/python-v3.13-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.31.0-red)
![LangChain](https://img.shields.io/badge/langchain-0.1.6-green)

## 🌟 Features

- 📤 **Upload PDF files** - Support for any PDF document
- 💬 **Interactive Chat** - Ask questions in natural language
- 🧠 **RAG Technology** - Retrieval-Augmented Generation for accurate answers
- ⚡ **Fast Responses** - Powered by Groq's lightning-fast LLM
- 🎨 **Beautiful UI** - Clean and intuitive interface

## 🛠️ Technologies Used

- **LangChain**: Framework for building LLM applications
- **Groq**: Ultra-fast LLM inference
- **FAISS**: Vector similarity search
- **Streamlit**: Web application framework
- **HuggingFace Embeddings**: Text embeddings model
- **PyPDF2**: PDF text extraction

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Groq API key ([Get it free here](https://console.groq.com))

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pdf-chat-rag.git
cd pdf-chat-rag
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

5. Open your browser at `http://localhost:8501`

## 📖 How to Use

1. Enter your Groq API key in the sidebar
2. Upload a PDF file using the file uploader
3. Wait for the PDF to be processed
4. Start asking questions about your PDF!

## 🏗️ Architecture

The application uses RAG (Retrieval-Augmented Generation) architecture:

1. **Document Loading**: PDF is loaded and text is extracted
2. **Text Chunking**: Document is split into manageable chunks
3. **Embedding**: Chunks are converted to vector embeddings
4. **Vector Storage**: Embeddings stored in FAISS for fast retrieval
5. **Query Processing**: User questions are embedded and relevant chunks retrieved
6. **Answer Generation**: LLM generates answers based on retrieved context

## 🔧 Configuration

You can modify these settings in `app.py`:

- `chunk_size`: Size of text chunks (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)
- `model_name`: Groq model to use (default: "mixtral-8x7b-32768")
- `k`: Number of chunks to retrieve (default: 3)

## 📝 Example Questions

- "What is the main topic of this document?"
- "Summarize the key points"
- "What does it say about [specific topic]?"
- "List the main conclusions"

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- LangChain for the amazing framework
- Groq for ultra-fast inference
- Streamlit for the beautiful UI framework

## 📧 Contact

Your Name - [@yourhandle](https://twitter.com/yourhandle)

Project Link: [https://github.com/yourusername/pdf-chat-rag](https://github.com/yourusername/pdf-chat-rag)

---

⭐ Star this repo if you find it helpful!