import boto3 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
import os 
import shutil
from botocore.exceptions import ClientError
from tqdm import tqdm 

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
    
    # Split the documents into chunks using recursive splitting
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


def generate_embeddings_and_store(rag_file_path,
                                  local_dir,
                                  model, 
                                  splitter_type='recursive', 
                                  chunk_size=2000, 
                                  chunk_overlap=300):
    """
    Generates embeddings for a given PDF document and stores them in a FAISS vector store on S3.

    Args:
        rag_file_path (str): The file path of the PDF document to process.
        local_dir (str): The local directory to save the vector store before uploading to S3.
        model (str): The HuggingFace model name to use for generating embeddings.
        splitter_type (str): The type of text splitter to use ('recursive', 'unstruct_basic', 'unstruct_by_title').
        chunk_size (int): The maximum size of each text chunk.
        chunk_overlap (int): The number of overlapping characters between chunks (only applicable for 'recursive' and 'unstruct_basic' splitters).
    """

    # Partition the PDF document based on the specified splitter type
    with tqdm(total=1, desc = f"Splitting documents with {splitter_type} splitter") as pbar:
        if splitter_type == 'recursive':
            docs = recursive_splitter(rag_file_path, chunk_size, chunk_overlap)
        elif splitter_type == 'unstruct_basic':
            docs = unstructured_basic_splitter(rag_file_path, chunk_size, chunk_overlap)
        elif splitter_type == 'unstruct_by_title':
            docs = unstructured_by_title_splitter(rag_file_path, chunk_size)
        else:
            raise ValueError("Invalid splitter type. Choose from 'recursive', 'unstruct_basic', or 'unstruct_by_title'.")
        pbar.update(1)
        
    # Get the HuggingFace embedding model
    embedding_model = get_huggingface_embedding_model(model)
    
    # Create embedding vector store
    knowledge_vector_db = FAISS.from_documents(
        tqdm(docs, desc="Embedding documents"), 
        embedding_model, 
        distance_strategy=DistanceStrategy.COSINE
    )

    # Save vector store locally
    save_vs_path = local_dir / "vector_stores" / model.replace("/", "-") / splitter_type
    knowledge_vector_db.save_local(str(save_vs_path))

    # Save vector store to S3
    s3_path = f"vector_stores/{model.replace('/', '-')}/{splitter_type}"
    upload_to_s3(save_vs_path, s3_path)

    # Remove local vector store after upload
    try:
        shutil.rmtree(save_vs_path)
    except OSError as e:
        print(f"Error deleting directory: {e}")


def upload_to_s3(local_folder, s3_path):
    """
    Uploads a local folder to S3.

    Args:
    local_folder (str): The local folder path to upload.
    s3_path (str): The S3 path to upload to.
    """
    
    try:
        s3 = boto3.client('s3')
    except ClientError as e:
        print(f"Error creating S3 client: {e}")
        return
    
    for root, dirs, files in os.walk(local_folder):
        # Get all local files
        for file in files:
            local_path = os.path.join(root, file)
            s3_file_path = os.path.join(s3_path, file)
        
            try:
                # Upload to S3
                s3.upload_file(local_path, S3_BUCKET_NAME, s3_file_path)
                print(f"Uploaded file from {local_path} to s3://{S3_BUCKET_NAME}/{s3_path}")
            except ClientError as e:
                print(f"Error uploading to S3: {e}")


def load_embedding_vector_store(model, splitter_type):
    """
    Load the FAISS embedding vector store from S3.

    Args:
        model (str): The HuggingFace embedding model name.    
        splitter_type (str): The splitter type.
    
    Returns:
        FAISS: The loaded vector store.
    """
    
    try:
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME)
    except ClientError as e:
        print(f"Error creating S3 client: {e}")
        return 
    
    # Get local path for the vector store 
    local_path, dir_exists = get_local_dir(model, splitter_type)
    
    # Get S3 keys for vector stores
    s3_keys = [
        ele['Key'] for ele in response['Contents']
        if f"vector_stores/{model.replace('/', '-')}/{splitter_type}/" in ele['Key']
    ]

    # Download from S3
    if not dir_exists:
        for key in s3_keys:
    
            # Skip directory key
            if key.endswith('/'):
                continue
    
            # Extract file name
            file_name = os.path.basename(key)
            
            try:
                s3.download_file(S3_BUCKET_NAME, key, f"{local_path}/{file_name}")
                print(f"Saved file from s3://{S3_BUCKET_NAME}/{key} at {local_path}/{file_name}.")
            except ClientError as e:
                print(f"Error downloading from S3: {e}")
                return 
            
    return FAISS.load_local(local_path, 
                            get_huggingface_embedding_model(model), 
                            allow_dangerous_deserialization=True)


def get_local_dir(model, splitter_type):
    """
    Get the local directory for the vector store.

    Args:
        model (str): The model name.
        splitter_type (str): The splitter type.

    Returns:
        tuple: A tuple containing the local path and a boolean indicating if the directory exists.
    """

    # Create local path to save vector store 
    try:
        base_dir = Path(__file__).parent.parent
    except NameError:
        base_dir = Path.cwd().parent 

    local_path = base_dir / "temp" / "vector_stores" / model.replace('/', '-') / splitter_type

    # Check if directory exists and is not empty
    if local_path.is_dir() and any(local_path.iterdir()):
        print("Loading locally available vector store.")
        return (local_path, True)
    else:
        # Create directory for vector store if it doesn't exist
        local_path.mkdir(parents=True, exist_ok=True)
        return (local_path, False)


def download_from_s3(s3_path, output_path):
    """
    Download a file from S3 to a local directory.

    Args:
        s3_path (str): The S3 path of the file.
        output_path (str): The local directory to save the file.
    """
    # Extract file name
    file_name = os.path.basename(s3_path)

    try:
        s3 = boto3.client('s3')
    except ClientError as e:
        print(f"Error creating S3 client: {e}")
        return

    try:
        s3.download_file(S3_BUCKET_NAME, s3_path, f"{output_path}/{file_name}")
        print(f"Saved file from s3://{S3_BUCKET_NAME}/{s3_path} at {output_path}.")
    except ClientError as e:
        print(f"Error downloading from S3: {e}")
        