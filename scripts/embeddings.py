import boto3 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
import os 

S3_BUCKET_NAME = "local-gov-ai-llm-benchmarking"

def recursive_splitter(file_path, chunk_size=2000, chunk_overlap=300):
    """
    Splits a PDF document into smaller chunks using recursive character text splitting.
    
    Returns:
        List[Document] List of documents (LangChain Document class) split into chunks.
    """
    # Load the PDF file
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Split the documents into chunks using recusrive splitting
    text_splitter = RecursiveCharacterTextSplitter(
                                                    chunk_size=chunk_size, 
                                                    chunk_overlap=chunk_overlap,
                                                    length_function=len
                                                    )
    return text_splitter.split_documents(documents)


def unstructured_basic_splitter(file_path, chunk_size=2000, chunk_overlap=300):
    """
    Splits a PDF document into smaller chunks using unstructured basic text splitting.
    
    Returns:
        List[Document] List of documents (LangChain Document class) split into chunks.
    """
    # Define the UnstructuredLoader with basic chunking strategy
    loader_basic = UnstructuredLoader(
                                file_path=file_path,
                                strategy="hi_res",
                                chunking_strategy = "basic",
                                max_characters=chunk_size,
                                new_after_n_chars=(chunk_size - 300),
                                include_orig_elements=True,
                                overlap=chunk_overlap,
                                overlap_all=True,
                                extract_images_from_text=False
    )
    
    docs_basic = []

    # Load the PDF document and split it into chunks
    for doc in loader_basic.lazy_load():
        docs_basic.append(doc)
    
    return docs_basic


def unstructured_by_title_splitter(file_path, chunk_size=2000):
    """
    Splits a PDF document into smaller chunks using unstructured by title text splitting.
    
    Returns:
        List[Document] List of documents (LangChain Document class) split into chunks.
    """
    # Unstructured partitioning with chunking by title
    loader_by_title = UnstructuredLoader(
                                        file_path=file_path,
                                        strategy="hi_res",
                                        chunking_strategy = "by_title",
                                        max_characters=chunk_size,
                                        include_orig_elements=True,
                                        combine_text_under_n_chars=(chunk_size - 300),
                                        extract_images_from_text=False
                                    )
    
    docs_by_title = []

    # Load the PDF document and split it into chunks
    for doc in loader_by_title.lazy_load():
        docs_by_title.append(doc)
    
    return docs_by_title


def get_huggingface_embedding_model(model):
    """Returns a HuggingFaceEmbeddings instance with the specified model.
    """
    if not model:
        raise ValueError("Model name must be provided.")
    else:
        return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={"normalize_embeddings": True} 
    )


def generate_embeddings_and_store(file_path, 
                                  model, splitter_type='recursive', 
                                  chunk_size=2000, chunk_overlap=300, ):

    # Partition the PDF document based on the specified splitter type
    if splitter_type == 'recursive':
        docs = recursive_splitter(file_path, chunk_size, chunk_overlap)
    elif splitter_type == 'unstruct_basic':
        docs = unstructured_basic_splitter(file_path, chunk_size, chunk_overlap)
    elif splitter_type == 'unstruct_by_title':
        docs = unstructured_by_title_splitter(file_path, chunk_size)
    else:
        raise ValueError("Invalid splitter type. Choose from 'recursive', 'unstruct_basic', or 'unstruct_by_title'.")
    
    # Get the HuggingFace embedding model
    embedding_model = get_huggingface_embedding_model(model)
    
    # Create vector store
    knowledge_vector_db = FAISS.from_documents(
        docs, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )

    # Save vector store locally
    base_dir = Path(__file__).parent.parent 
    save_vs_path = base_dir / "data" / "vector_stores" / model / splitter_type
    knowledge_vector_db.save_local(str(save_vs_path))

    # Save vector store to S3
    upload_to_s3(save_vs_path)

    # Remove local vector store after upload
    try:
        os.rmdir(save_vs_path)
    except OSError as e:
        print(f"Error deleting directory: {e}")


def upload_to_s3(local_path):
    
    try:
        s3 = boto3.client('s3')
    except Exception as e:
        print(f"Error creating S3 client: {e}")
        return
    
    # Compute S3 path 
    parts = Path(local_path).parts
    s3_path = Path(*parts[2:])
    
    # Upload to S3
    try:
        s3.upload_file(local_path, S3_BUCKET_NAME, s3_path)
        print(f"Uploaded vector store from {local_path} to s3://{S3_BUCKET_NAME}/{s3_path}")
    except Exception as e:
        raise(f"Error uploading to S3: {e}")


def load_embedding_vector_store(model, splitter_type):
    """
    Load the FAISS embedding vector store from S3.
    
    Returns:
        FAISS: The loaded vector store.
    """
    
    try:
        s3 = boto3.client('s3')
    except Exception as e:
        print(f"Error creating S3 client: {e}")
        return
    
    # Define local path to save vector store and S3 path for retrieval
    base_dir = Path(__file__).parent.parent 
    local_path = base_dir / "data" / "vector_stores" / model / splitter_type
    s3_path = f"vector_stores/{model}/{splitter_type}"
    
    # Download from S3
    try:
        s3.download_file(S3_BUCKET_NAME, s3_path, str(local_path))
        print(f"Loaded vector store from s3://{S3_BUCKET_NAME}/{s3_path} at {local_path}.")
    except Exception as e:
        raise(f"Error downloading from S3: {e}")
        
    return FAISS.load_local(str(local_path), 
                            HuggingFaceEmbeddings(), 
                            allow_dangerous_deserialization=True)