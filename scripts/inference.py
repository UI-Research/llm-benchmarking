import boto3 
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_aws import BedrockLLM
from langchain.vectorstores import FAISS

import json 

QUERY_PROMPT_TEMPLATE = """\
H:
Answer the question based on the provided context.
{context}
Question: {question}
A:
"""

def get_inference_profile_arn(model):
    """
    Get the inference profile ARN for the specified model.
    
    Args:
        model (str): The name of the model's inference profile.
        
    Returns:
        str: The ARN of the inference profile.
    """
    try:
        client = boto3.client(service_name="bedrock", 
                              region_name="us-east-1")
        response = client.list_inference_profiles()
    except Exception as e:
        print(f"Error accessing inference profiles using Bedrock client: {e}")
        return None
    
    inf_profile = next( ele['inferenceProfileArn'] 
                       for ele in response['inferenceProfileSummaries'] 
                       if ele['inferenceProfileName'] == model)
    
    return inf_profile


def retrieval_qa_chain(model, provider_name, embedding_vs):
    """Create a RetrievalQA chain using the specified model and 
    embedding vector store. 
    
    Args:
        model (str): The name of the model's inference profile.
        provider_name (str): The name of the provider for the LLM.
        embedding_vs (FAISS): The FAISS vector store containing embeddings.
    
    Returns:
        RetrievalQA: A LangChain RetrievalQA chain configured with the 
        specified model and vector store.
    """
    # Connect to bedrock runtime
    try:    
        bedrock_client = boto3.client(service_name="bedrock-runtime", 
                                      region_name="us-east-1")
    except Exception as e:
        print(f"Error creating Bedrock client: {e}")
        return None
    
    # Get the model's inference profile
    inf_profile = get_inference_profile_arn(model)
    
    # Define the LLM model using the BedrockLLM class
    llm_model = BedrockLLM(model_id=inf_profile, 
                           client=bedrock_client, 
                           provider=provider_name)

    # Define the QA chain with a custom prompt template
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_model,
        retriever=embedding_vs.as_retriever(search_kwargs={'k': 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PromptTemplate.from_template(QUERY_PROMPT_TEMPLATE)}
    )

    return qa_chain


def run_inference(model, embedding_vs, query):
    
    # Retrieve contect from the vector store based on the user query
    retrieved_context = embedding_vs.similarity_search(query, k=5)
    
    if not retrieved_context:
        print("No relevant context found for the query.")
        return
    
    # Get the model's inference profile
    inf_profile = get_inference_profile_arn(model)
    
    # Connect to Bedrock runtime
    try:
        client = boto3.client(service_name="bedrock-runtime", 
                              region_name="us-east-1")
    except Exception as e:
        print(f"Error creating Bedrock client: {e}")
        return
    
    # Invoke the model with the retrieved context and user query
    try:
        response = client.invoke_model(
            modelId=inf_profile,
            body=json.dumps({
                "messages": [
                    {"role": "system", "content": "Answer the question based on the provided context."},
                    {"role": "user", "content": f"{retrieved_context[0]}\nQuestion: {query}"}
                ]
            }),
            contentType="application/json"
        )
    except Exception as e:
        print(f"Error invoking model: {e}")
    
    return response
