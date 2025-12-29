import boto3 
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_aws import BedrockLLM
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
import pandas as pd
import json 
from botocore.exceptions import ClientError
from botocore.config import Config
from prompts import SYSTEM_PROMPT, PROMPT
from pathlib import Path
import datetime
import sys
import os
from dotenv import load_dotenv

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
    queries['context_metadata'] = None

    for i in range(k):
        queries[f'context_doc_{i+1}'] = None

    # For each query
    for idx, row in queries.iterrows():

        # Retrieve top-k context from the vector store based on the user query
        top_k_context = get_top_k_context_metadata(embedding_vs, row['question'], k)
       
        # Combine question and context to create user message
        context_docs = ""
        context_metadata = {}
        
        for i, doc in enumerate(top_k_context):
            # Combine retrieved context
            context_docs += f"Context {i+1}: {doc.page_content} \n"
            # Collect context metadata
            context_metadata[f"Context {i+1}"] = doc.metadata
            # Add individual context doc to a column
            queries.at[idx, f'context_doc_{i+1}'] = doc.page_content.strip()
        
        queries.at[idx, 'query_w_context'] = f"Question: {row['question']} \nContext:\n\n {context_docs}"
        queries.at[idx, 'all_context_docs'] = context_docs
        queries.at[idx, 'context_metadata'] = context_metadata

    return queries


def get_top_k_context_metadata(emb_vs, query, k):

    # Retrieve context from the vector store based on the user query
    top_k_context = emb_vs.similarity_search(query, k)

    return top_k_context
    

def run_conversation(model, q_type, emb_name, emb_vs, folder_path, k=6, n_iter=3):
    
    # Connect to Bedrock runtime
    try:
        config = Config(
            read_timeout=120    # Timeout for reading data from the connection
        )
        bedrock_client = boto3.client(service_name="bedrock-runtime", 
                              region_name="us-east-1", config=config)
    except ClientError as e:
        print(f"Error creating Bedrock client: {e}")
        return
        
    # Get the model's inference profile
    inf_profile = get_inference_profile_arn(model)

    # Choose model id based on if inference profile is required or not
    model_id = model if inf_profile is None else inf_profile
    
    # Get user questions and respective context
    queries = get_user_queries_with_context(q_type, emb_vs, k)
    
    # Define columns 
    queries['model'] = model
    queries['execution_date'] = datetime.date.today()
    queries['embedding_model'] = emb_name
    
    # Set up system prompt
    system_prompt = [{"text": SYSTEM_PROMPT}]

    all_iterations = pd.DataFrame()

    for i in range(n_iter):
        # Clear response and update iteration for current run
        queries['iteration'] = i+1
        queries['response'] = None
        queries['reasoning'] = None

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

                if "read timeout" in str(e).lower():
                    print("Retrying conversation with a different AWS region.")

                    try: 
                        # Set bedrock client with different region
                        bedrock_client = boto3.client(service_name="bedrock-runtime", 
                                  region_name="us-east-2", config=config)
                        
                        # Get model response for current message using Converse API
                        response = generate_conversation(bedrock_client, model_id, messages, system_prompt)
                        
                    except Exception as er:
                        print(f"Retry failed: {er}")
                        return None
                else:  
                    return
    
            # Extract model's response and append to history
            output_message = response['output']['message']
            messages.append(output_message)
    
            # Save model's response
            if "gpt" in model_id:
                queries.at[idx, 'response'] = output_message['content'][1]['text']
                queries.at[idx, 'reasoning'] = output_message['content'][0]['reasoningContent']['reasoningText']['text']
            else:
                queries.at[idx, 'response'] = output_message['content'][0]['text']
    
            # Print current conversation 
            # print(f"Role: user\nQuestion: {row['question']}")
            # print(f"Role: assistant\nResponse: {output_message['content'][0]['text']}")

        # Store copy of current iteration results
        all_iterations = pd.concat([all_iterations, queries.copy()], ignore_index=True)

    current_date = datetime.date.today().strftime("%Y-%m-%d")
    file_path = folder_path / f"output_{q_type}_{current_date}.csv"
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
    

def retrieval_qa_chain_openai(model, provider_name, embedding_vs):
    """Create a RetrievalQA chain using the specified model and 
    embedding vector store. 
    
    Args:
        model (str): Name of the OpenAI model.
        embedding_vs (FAISS): The FAISS vector store containing embeddings.
    
    Returns:
        RetrievalQA: A LangChain RetrievalQA chain configured with the 
        specified OpenAI model and vector store.
    """
    
    # Define the LLM model using the BedrockLLM class
    llm_model = ChatOpenAI(temperature=0, model_name="gpt-3.5")

    # Return the QA chain with a custom prompt template
    return RetrievalQA.from_chain_type(
        llm=llm_model,
        retriever=embedding_vs.as_retriever(search_kwargs={'k': 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )


def rag_with_openai(model_id, q_type, emb_name, emb_vs, folder_path, k=6, n_iter=3):

    # Load OpenAI API key and initialize a model
    load_dotenv(override=True)
    
    llm = ChatOpenAI(
        model=model_id,
    )
    
    # Set up system prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}")
    ])

    # Create question and answer chain
    qa_chain = qa_prompt | llm

    # Get user questions and respective context
    queries = get_user_queries_with_context(q_type, emb_vs, k)
    
    # Define columns 
    queries['model'] = model_id
    queries['execution_date'] = datetime.date.today()
    queries['embedding_model'] = emb_name
    
    all_iterations = pd.DataFrame()

    for i in range(n_iter):
        # Clear response and update iteration for current run
        queries['iteration'] = i+1
        queries['response'] = None
        queries['reasoning'] = None

        print(f"Conversation iteration {i+1}")
        # Clear messages for current iteration
        chat_history = []
        
        for idx, row in queries.iterrows():
            try: 
                # Get OpenAI model response for current message 
                response = qa_chain.invoke({
                                "context": row['all_context_docs'],
                                "chat_history": chat_history,
                                "question": row['question']
                            })
            except Exception as e:
                print(f"Error occurred while generating conversation with OpenAI: {e}")
                return

            # Add current query to history
            chat_history.append(HumanMessage(content = row['question']))
            
            # Extract model's response and append to history
            chat_history.append(AIMessage(content = response.content))
    
            # Save model's response
            queries.at[idx, 'response'] = response.content
    
        # Store copy of current iteration results
        all_iterations = pd.concat([all_iterations, queries.copy()], ignore_index=True)

    current_date = datetime.date.today().strftime("%Y-%m-%d")
    file_path = folder_path / f"output_{q_type}_{current_date}.csv"
    all_iterations.to_csv(file_path)


def get_test_queries_with_context(queries, embedding_vs, k):

    # Define columns
    queries['context_metadata'] = None

    for i in range(k):
        queries[f'context_doc_{i+1}'] = None

    # For each query
    for idx, row in queries.iterrows():

        # Retrieve top-k context from the vector store based on the user query
        top_k_context = get_top_k_context_metadata(embedding_vs, row['test_query'], k)

        context_metadata = {}
        
        for i, doc in enumerate(top_k_context):
            # Collect context metadata
            context_metadata[f"Context {i+1}"] = doc.metadata
            # Add individual context doc to a column
            queries.at[idx, f'context_doc_{i+1}'] = doc.page_content.strip()
        
        queries.at[idx, 'context_metadata'] = context_metadata

    return queries
    