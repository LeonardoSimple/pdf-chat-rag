import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from PyPDF2 import PdfReader

# Page config
st.set_page_config(
    page_title="Chat with PDF", 
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">📄 Chat with Your PDF</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload any PDF and ask questions - powered by RAG & LangChain</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password", help="Get your free API key from console.groq.com")
    
    if groq_api_key:
        st.success("✅ API Key configured")
    
    st.markdown("---")
    st.markdown("### 📚 How to use:")
    st.markdown("""
    1. Enter your Groq API key
    2. Upload a PDF file
    3. Wait for processing
    4. Ask questions!
    """)
    
    st.markdown("---")
    st.markdown("### 🔗 Quick Links:")
    st.markdown("[Get Groq API Key](https://console.groq.com)")
    st.markdown("[GitHub Repo](https://github.com/aliabdm)")
    
    st.markdown("---")
    st.markdown("### 🛠️ Built with:")
    st.markdown("""
    • **LangChain** - RAG Framework
    • **Groq** - Fast LLM Inference
    • **FAISS** - Vector Search
    • **Streamlit** - UI Framework
    """)
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Developer")
    st.markdown("**Mohammad Ali Abdul Wahed**")
    st.markdown("Developer • Creator • OSS Enthusiast")
    
    st.markdown("##### Connect with me:")
    st.markdown("🔗 [LinkedIn](https://www.linkedin.com/in/mohammad-ali-abdul-wahed-1533b9171/)")
    st.markdown("💻 [GitHub](https://github.com/aliabdm)")
    st.markdown("✍️ [Medium](https://medium.com/@aliabdm)")
    st.markdown("📝 [Dev.to](https://dev.to/maliano63717738)")
    
    st.markdown("---")
    st.markdown("*© 2025 - Open Source & Free to Use*")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

# Main content
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    uploaded_file = st.file_uploader(
        "📎 Choose a PDF file", 
        type=['pdf'],
        help="Upload any PDF document to start chatting"
    )

# Process PDF
if uploaded_file and groq_api_key:
    file_name = uploaded_file.name
    
    # Check if we need to reprocess
    if st.session_state.processed_file != file_name:
        st.session_state.vector_store = None
        st.session_state.messages = []
        st.session_state.processed_file = file_name
    
    if st.session_state.vector_store is None:
        with st.spinner("🔄 Processing your PDF... This may take a minute."):
            try:
                # Extract text from PDF
                pdf_reader = PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                
                if not text.strip():
                    st.error("❌ Could not extract text from PDF. Make sure it's not scanned or image-based.")
                    st.stop()
                
                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(text)
                
                # Create embeddings and vector store
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
                
                st.success(f"✅ Successfully processed **{file_name}**! ({len(chunks)} chunks created)")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")
                st.stop()
    
    # Chat interface
    st.markdown("---")
    st.markdown("### 💬 Chat with your PDF")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if question := st.chat_input("Ask anything about your PDF..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # Get relevant documents
                    docs = st.session_state.vector_store.similarity_search(question, k=3)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    # Create prompt
                    prompt_template = """You are a helpful assistant that answers questions based on the provided context.
                    
Context from the PDF:
{context}

Question: {question}

Please provide a detailed and accurate answer based only on the context above. If the answer cannot be found in the context, say so."""
                    
                    prompt = ChatPromptTemplate.from_template(prompt_template)
                    
                    # Setup LLM
                    llm = ChatGroq(
                        temperature=0,
                        groq_api_key=groq_api_key,
                        model_name="llama-3.3-70b-versatile"
                    )
                    
                    # Create chain
                    chain = (
                        {"context": lambda x: context, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                    
                    # Get answer
                    answer = chain.invoke(question)
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

elif not groq_api_key:
    st.warning("⚠️ Please enter your Groq API key in the sidebar to continue")
    st.info("👉 You can get a free API key from [console.groq.com](https://console.groq.com)")
else:
    st.info("👆 Upload a PDF file to start chatting!")
    
    # Show example
    with st.expander("📖 See example use cases"):
        st.markdown("""
        **What you can do:**
        - 📊 Analyze research papers
        - 📝 Summarize long documents
        - 🔍 Find specific information quickly
        - ❓ Ask questions about the content
        - 💡 Get explanations of complex topics
        
        **Example questions:**
        - "What is the main topic of this document?"
        - "Summarize the key points"
        - "What does it say about [specific topic]?"
        - "List the main conclusions"
        """)