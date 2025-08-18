import embeddings as emb
import boto3
from pathlib import Path
from tqdm import tqdm

if __name__ == "__main__":
    
    # Download zoning code for RAG
    base_dir = Path(__file__).parent.parent 
    local_file_dir = base_dir / "temp" 
    local_file_dir.mkdir(exist_ok=True)
    
    emb.download_from_s3(s3_path="Minneapolis_MN_Code_of_Ordinances.pdf", output_path=local_file_dir)
    
    # Define document chunk size and overlap
    chunk_size_value = 2000
    chunk_overlap = 300
    
    # HuggingFace embedding models
    embedding_models = ["intfloat/multilingual-e5-large-instruct", "intfloat/e5-small-v2"]
    
    # HuggingFace document splitter types
    splitters = ["recursive", "unstruct_basic", "unstruct_by_title"]
    
    # Generate embeddings and store them in AWS S3
    for emb_model in embedding_models:
        for splitter in splitters:
            try:
                emb.generate_embeddings_and_store(
                    rag_file_path=f"{local_file_dir}/Minneapolis_MN_Code_of_Ordinances.pdf", 
                    local_dir=local_file_dir,
                    model=emb_model, 
                    splitter_type=splitter, 
                    chunk_size=chunk_size_value, 
                    chunk_overlap=chunk_overlap
                )
                print(f"Generated embeddings for {emb_model} with splitter type '{splitter}'")
            except Exception as e:
                print(f"Error for splitter {splitter}: {e}")
        
