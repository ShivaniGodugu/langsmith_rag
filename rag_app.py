import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# ===== SET YOUR API KEY HERE =====
# IMPORTANT: Replace this with your actual full API key
# Get it from: https://platform.openai.com/api-keys
OPENAI_API_KEY = "sk-proj-your-full-actual-key-here"  # <-- REPLACE THIS

# Set the key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Verify key is set
print(f"API Key set: {'✅' if os.getenv('OPENAI_API_KEY') else '❌'}")

def test_api_key():
    """Test if API key works"""
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        response = llm.invoke("Say 'API key works'")
        print("✅ API key test successful!")
        print(f"Response: {response.content}")
        return True
    except Exception as e:
        print(f"❌ API key test failed: {e}")
        return False

def load_docs():
    """Load PDF document"""
    file_path = "docs/guide.pdf"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []
    
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        print(f"✅ Loaded {len(docs)} pages from {file_path}")
        return docs
    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        return []

def build_vectorstore(docs, use_smaller_chunks=True):
    """Create vector store with smaller chunks for faster processing"""
    if not docs:
        return None
    
    # Use smaller chunks for faster processing
    chunk_size = 500 if use_smaller_chunks else 800
    chunk_overlap = 50
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    print(f"Splitting {len(docs)} documents into chunks...")
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks")
    
    # Limit chunks for faster processing (remove this line to process all)
    if len(chunks) > 500:
        print(f"⚠️ Limiting to first 500 chunks for faster processing")
        chunks = chunks[:500]
    
    try:
        print("Creating embeddings (this may take a minute)...")
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(
            chunks, 
            embedding=embeddings, 
            collection_name="demo-rag"
        )
        print("✅ Vector store created successfully")
        return vectordb
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        return None

def build_rag_chain(vectordb):
    """Build RAG chain"""
    if not vectordb:
        return None
    
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Use the following context to answer the question.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        "Context:\n{context}\n\nQuestion: {input}"
    )

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    return rag_chain

def get_answer(question: str) -> str:
    """Get answer from RAG system"""
    if not question:
        return "Please provide a question."
    
    print(f"\n📝 Question: {question}")
    
    # Test API key first
    if not test_api_key():
        return "❌ API key is invalid. Please check your API key and try again."
    
    # Load documents
    docs = load_docs()
    if not docs:
        return "❌ No documents found. Please add 'docs/guide.pdf' file."
    
    # Create vector store
    vectordb = build_vectorstore(docs)
    if not vectordb:
        return "❌ Failed to create vector store."
    
    # Build RAG chain
    rag_chain = build_rag_chain(vectordb)
    if not rag_chain:
        return "❌ Failed to build RAG chain."
    
    # Get answer
    try:
        print("🤔 Getting answer...")
        result = rag_chain.invoke({"input": question})
        return result["answer"]
    except Exception as e:
        return f"❌ Error getting answer: {str(e)}"

# Main execution
if __name__ == "__main__":
    print("=== RAG System ===")
    
    # Check if API key is set
    if OPENAI_API_KEY == 'your-key-here':
        print("⚠️  WARNING: You haven't set your actual API key!")
        print("Please replace OPENAI_API_KEY with your real key from https://platform.openai.com/api-keys")
    else:
        print("API key configured, testing connection...")
        q = input("\nAsk a question: ")
        answer = get_answer(q)
        print(f"\n💡 Answer: {answer}")