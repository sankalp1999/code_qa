from flask import Flask, render_template, request, session
import os
import sys
import lancedb
import cohere
import pandas as pd
import re
import redis
import uuid
import logging
import markdown
from openai import OpenAI
from prompts import (
    HYDE_SYSTEM_PROMPT,
    HYDE_V2_SYSTEM_PROMPT,
    REFERENCES_SYSTEM_PROMPT,
    CHAT_SYSTEM_PROMPT  
)

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

app = Flask(__name__)

@app.template_filter('markdown')
def markdown_filter(text):
    return markdown.markdown(text, extensions=['fenced_code', 'tables'])

app.secret_key = os.urandom(24)

# Redis connection
redis_client = redis.Redis(host='localhost', port=6379, db=0)
 

# command line args here
if len(sys.argv) != 3:
    print("Usage: python app.py <language> <codebase_path>")
    sys.exit(1)

language = sys.argv[1]
codebase_path = sys.argv[2]

# ix = open_existing_index(codebase_path)

normalized_path = os.path.normpath(os.path.abspath(codebase_path))
    

codebase_folder_name = os.path.basename(normalized_path)

# lancedb connection
uri = "database"
db = lancedb.connect(uri)

method_table = db.open_table(codebase_folder_name + "_method")
class_table = db.open_table(codebase_folder_name + "_class")


# Replace Groq and Anthropic client setup with OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Cohere API setup
cohere_key = os.environ.get("COHERE_API_KEY")
co = cohere.Client(cohere_key)


# Replace groq_hyde function
def openai_hyde(query):
    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": HYDE_SYSTEM_PROMPT.format(language=language)
            },
            {
                "role": "user",
                "content": f"Help predict the answer to the query: {query} in the programming language: {language}.",
            }
        ]
    )
    return chat_completion.choices[0].message.content

# Replace groq_hyde_v2 function
def openai_hyde_v2(query, temp_context, hyde_query):
    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": HYDE_V2_SYSTEM_PROMPT.format(language=language, query=query, temp_context=temp_context)
            },
            {
                "role": "user",
                "content": f"Predict the answer to the query: {hyde_query} in the context of {language}.",
            }
        ]
    )
    return chat_completion.choices[0].message.content

def openai_query_for_references(query, context):
    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": REFERENCES_SYSTEM_PROMPT.format(query=query, context=context)
            },
            {
                "role": "user",
                "content": f"<query>{query}</query>",
            }
        ]
    )
    return chat_completion.choices[0].message.content

def openai_chat(query, context):
    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": CHAT_SYSTEM_PROMPT.format(context=context)
            },
            {
                "role": "user",
                "content": query,
            }
        ]
    )
    return chat_completion.choices[0].message.content

def process_input(input_text):
    processed_text = input_text.replace('\n', ' ').replace('\t', ' ')
    processed_text = re.sub(r'\s+', ' ', processed_text)
    processed_text = processed_text.strip()
    
    return processed_text

def generate_context(query):
    hyde_query = openai_hyde(query)

    method_docs = method_table.search(hyde_query).limit(5).to_pandas()
    class_docs = class_table.search(hyde_query).limit(5).to_pandas()

    temp_context = '\n'.join(method_docs['code'] + '\n'.join(class_docs['class_info']) )

    # can switch to 70b for this if can reduce num of tokens
    hyde_query_v2 = openai_query_for_references(query, temp_context)

    logging.info("-query_v2-")
    logging.info(hyde_query_v2)

    method_docs = method_table.search(hyde_query_v2).limit(5).to_pandas()
    class_docs = class_table.search(hyde_query_v2).limit(5).to_pandas()

    # logging.info("---v2---")
    # logging.info(method_docs['code'].tolist())
    # logging.info(class_docs['class_info'].tolist())
    # logging.info("-------")

    method_results = co.rerank(query=hyde_query_v2, documents=method_docs['code'].tolist(), top_n=5, model='rerank-english-v3.0')
    class_results = co.rerank(query=hyde_query_v2, documents=class_docs['class_info'].tolist(), top_n=5, model='rerank-english-v3.0')

    selected_method_rows = [method_docs.iloc[hit.index] for hit in method_results.results]
    selected_class_rows = [class_docs.iloc[hit.index] for hit in class_results.results]

    top_k_reranked_method_results_df = pd.DataFrame(selected_method_rows)
    top_k_reranked_class_results_df = pd.DataFrame(selected_class_rows)

    top_k_reranked_method_results_df.reset_index(drop=True, inplace=True)
    top_k_reranked_class_results_df.reset_index(drop=True, inplace=True)

    top_3_methods = top_k_reranked_method_results_df.iloc[:3]
    methods_combined = "\n\n".join(f"File: {row['file_path']}\nCode:\n{row['code']}" for index, row in top_3_methods.iterrows())

    top_3_classes = top_k_reranked_class_results_df.iloc[:3]
    classes_combined = "\n\n".join(f"File: {row['file_path']}\nClass Info:\n{row['class_info']} References: \n{row['references']}  \n END OF ROW {index}" for index, row in top_3_classes.iterrows())


    print("----------------")
    print(classes_combined)
    print("LENGTH:", len(classes_combined))
    print("----------------")
    print("Context generation is complete.")

    print("METHODS COMBINED")
    print(methods_combined)
    print("----METHODS END----")

    return methods_combined + "\n below is class or constructor related code \n" + classes_combined

@app.route('/', methods=['GET', 'POST'])
def home():
    results = None
    if request.method == 'POST':
        query = request.form['query']
        action = request.form['action']
        
        user_id = session.get('user_id')
        if user_id is None:
            user_id = str(uuid.uuid4())
            session['user_id'] = user_id

        if action == 'Context':
            ## Context generation
            context = generate_context(query)


            logging.info("-----context------")
            logging.info(len(process_input(context)))
            logging.info(context)
            logging.info("-----context_end-----")


        elif action == 'Chat':


            # retrieve context
            redis_key = f"user:{user_id}:chat_history"
            context = redis_client.get(redis_key)
            print("INSIDE CHAT CONTEXT:", context)
            if context is None:
                context = ""
            else:
                print("found context")
                context = context.decode()
            
        
        response = openai_chat(query, context[:10000]) # token rate limit is problematic

        combined_response = f"Query: {query} \n\n Response: {response}"

        redis_key = f"user:{user_id}:responses"
        redis_client.rpush(redis_key, combined_response)

        # Update chat history in Redis
        new_chat_history = (context + f"\nQuery: {query}\nResponse: {response}").strip()
        redis_client.set(f"user:{user_id}:chat_history", new_chat_history)

        # Retrieve the last 3 responses for the current user from Redis
        responses = redis_client.lrange(redis_key, -3, -1)
        responses = [response.decode() for response in responses]

        results = {
            'response': response,
            'responses': responses
        }

    return render_template('query_form.html', results=results)

if __name__ == "__main__":
    app.run(debug=True)
