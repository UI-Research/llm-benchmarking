import boto3 
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_aws import BedrockLLM
from langchain.vectorstores import FAISS
import pandas as pd
import json 
from botocore.exceptions import ClientError
from prompts import SYSTEM_PROMPT, PROMPT
from pathlib import Path
import datetime

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
    except ClientError as e:
        print(f"Error accessing inference profiles using Bedrock client: {e}")
        return 
    
    inf_profile = next( 
        (
            ele['inferenceProfileArn'] 
            for ele in response['inferenceProfileSummaries'] 
            if ele['inferenceProfileName'] == model), 
        None
    )
    
    return inf_profile

def generate_conversation(bedrock_client, model_id, messages, system_prompt):

    try: 
        response = bedrock_client.converse(
            modelId=model_id,
            messages=messages,
            system=system_prompt
        )
    except ClientError as e:
        print(f"Error invoking Converse API: {e}")
        return

    return response


def get_queries(q_type):

    # Read file containing all zoning related queries
    file_path = Path.cwd().parent / "temp" / "zoning-code-questions.csv"
    all_queries = pd.read_csv(file_path)

    # Filter queries for relevant question type
    queries = all_queries.loc[all_queries['index'].str.contains(q_type)]

    return queries


def get_user_queries_with_context(q_type, embedding_vs, k):

    # Get user queries based on question type
    queries = get_queries(q_type)

    # Define columns
    queries['query_w_context'] = None
    queries['context_docs'] = None
    queries['context_metadata'] = None

    for idx, row in queries.iterrows():

        # Retrieve context from the vector store based on the user query
        context_docs, context_metadata = get_context_metadata(embedding_vs, row['question'], k)
       
        # Combine question and context to create user message
        queries.at[idx, 'query_w_context'] = f"Question: {row['question']} \nContext:\n {context_docs}"
        queries.at[idx, 'context_docs'] = context_docs
        queries.at[idx, 'context_metadata'] = context_metadata

    return queries


def get_context_metadata(emb_vs, query, k):

    # Retrieve context from the vector store based on the user query
    top_k_context = emb_vs.similarity_search(query, k)
    
    context_docs = ""
    context_metadata = {}
    
    for i, doc in enumerate(top_k_context):
        # Combine retrieved context
        context_docs = context_docs + f"Context {i+1}: {doc.page_content} \n"

        # Collect context metadata
        context_metadata[f"Context {i+1}"] = doc.metadata

    return (context_docs, context_metadata)
    

def run_conversation(model, q_type, emb_vs, folder_path, k=3, n_iter=3):
    
    # Connect to Bedrock runtime
    try:
        bedrock_client = boto3.client(service_name="bedrock-runtime", 
                              region_name="us-east-1")
    except ClientError as e:
        print(f"Error creating Bedrock client: {e}")
        return
        
    # Get the model's inference profile
    inf_profile = get_inference_profile_arn(model)

    # Choose model id based on if inference profile is required or not
    if inf_profile is None:
        model_id = model
    else:
        model_id = inf_profile

    # Get user questions and respective context
    queries = get_user_queries_with_context(q_type, emb_vs, k)
    
    # Define columns 
    queries['model'] = model
    queries['execution_date'] = datetime.date.today()
    
    # Set up system prompt
    system_prompt = [{"text": SYSTEM_PROMPT}]

    all_iterations = pd.DataFrame()

    for i in range(n_iter):
        # Clear response and update iteration for current run
        queries['response'] = None
        queries['iteration'] = i+1

        print(f"Conversation iteration {i+1}")
        # Clear messages for current iteration
        messages = []
        
        for idx, row in queries.iterrows():
    
            # Add current message to history
            current_message = {
                "role": "user",
                "content": [{"text": row['query_w_context']}]
            }
            messages.append(current_message)
            
            try: 
                # Get model response for current message using Converse API
                response = generate_conversation(bedrock_client, model_id, messages, system_prompt)
            except Exception as e:
                print(f"Error occurred while generating conversation with Converse API: {e}")
                return
    
            # Extract model's response and append to history
            output_message = response['output']['message']
            messages.append(output_message)
    
            # Save model's response
            queries.at[idx, 'response'] = output_message['content'][0]['text']
    
            # Print current conversation 
            # print(f"Role: user\nQuestion: {row['question']}")
            # print(f"Role: assistant\nResponse: {output_message['content'][0]['text']}")

        # Store copy of current iteration results
        all_iterations = pd.concat([all_iterations, queries.copy()], ignore_index=True)

    file_path = folder_path / f"output_{q_type}.csv"
    all_iterations.to_csv(file_path)


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
    except ClientError as e:
        print(f"Error creating Bedrock client: {e}")
        return 
    
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
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain


def run_inference(model, embedding_vs, query):
    
    # Retrieve contect from the vector store based on the user query
    retrieved_context = embedding_vs.similarity_search(query, k=5)
    
    if not retrieved_context:
        print("No relevant context found for the query.")
        return None
    
    # Connect to Bedrock runtime
    try:
        client = boto3.client(service_name="bedrock-runtime", 
                              region_name="us-east-1")
    except ClientError as e:
        print(f"Error creating Bedrock client: {e}")
        return 

    # Get the model's inference profile
    inf_profile = get_inference_profile_arn(model)
    
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
    except ClientError as e:
        print(f"Error invoking model: {e}")
        return
    
    return response



    

    
    

    
    