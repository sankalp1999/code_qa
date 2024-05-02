from flask import Flask, render_template, request, session
import os
import sys
import lancedb
import cohere
import pandas as pd
from anthropic import Anthropic
import re
from fuzzy_search import open_existing_index, index_codebase, search_and_fetch_lines
import redis
import uuid
import logging


logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

app = Flask(__name__)

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


# Groq API setup
from groq import Groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Cohere API setup
cohere_key = os.environ.get("COHERE_API_KEY")
co = cohere.Client(cohere_key)


anthropic_client = Anthropic(
    # This is the default and can be omitted
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

# For Hyde, use llama70b for better reasoning and code
def groq_hyde(query):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f'''You are a software engineer who specializes in the programming language: {language}. 
                Predict the code for the query that can be the answer query provided in input.
                Think step by step. Try to be concise.
                If the question is a general one, then try to include name of relevant docs like README.md or config files that may contain info.
                Output format: Only the new query, no additional text'''
            },
            {
                "role": "user",
                "content": f"Help predict the answer to the query: {query} in the programming language: {language}.",
            }
        ],
        model="llama3-70b-8192",
    )
    return chat_completion.choices[0].message.content

def groq_hyde_v2(query, temp_context, hyde_query):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f'''You are a software engineer specializing in {language}. Improve the original query: {query} using the provided context: {temp_context}. 
                - If code-related, include relevant code snippets with specific method names and keywords.
                - If general, mention relevant files like README.md. 
                - If about a specific method, predict its implementation and suggest up-to-date libraries. 
                Keep the new query descriptive yet concise, focusing on expanding it with additional code-related keywords and details to better predict the answer.
                output format: just provide the query, do not add additional text.
                '''
            },
            {
                "role": "user",
                "content": f"Predict the answer to the query: {query} in the context of {language}.",
            }
        ],
        model="llama3-70b-8192",
    )
    return chat_completion.choices[0].message.content



def groq_query_for_references(query, context):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f'''Given the <query>{query}</query> and <context>{context}</context> , 
                1. Frame a concise query with help of provided context focusing on keywords that may help to answer the query, especially words not present in context.
                You may mention README.md. You may add answer keywords based on your own knowledge to the query or relevant keywords from context.
                For output, just provide the query, no additional text.
                Output format: 
                <query> new query here </query>
                '''
            },
            {
                "role": "user",
                "content": f"<query> {query} </query>",
            }
        ],
        model="llama3-70b-8192",
    )
    return chat_completion.choices[0].message.content


# For chat, changing to anthropic for higher context
def groq_chat(query, context):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"You are a software engineer. Using your knowledge and given the following <context> {context} </context, answer user's queries. Highlight particular code blocks, method names, class names."
            },
            {
                "role": "user",
                "content": f"{query}",
            }
        ],
        model="llama3-70b-8192",
    )
    return chat_completion.choices[0].message.content

def anthropic_chat(query, context):

    message = anthropic_client.messages.create(
    max_tokens=4096,
    system = f"You are a software engineer. Using your knowledge and given the following context:{context}, explain user's queries. Highlight particular code blocks, method names, class names. Be descriptive.",
    messages=[
        {
            "role": "user",
            "content": f"{query}",
        }
    ],
    model="claude-3-haiku-20240307",
    )
    return message.content[0].text

def anthropic_references(query, references):
    system = f'''Given the $query and <references> {references} </references>, 
                1. with help of references provided, grab relevant info like documentation, code snippet, method name, class name that look relevant for answering the query
                2. predict a better query under 4 lines with proper names and keywords with help of context which might look similar to answer to original query. try your best even if you are not confident.
                Output format: 
                <info> additional info here </info> 
                <query> new query here </query>
                '''
    message = anthropic_client.messages.create(
    max_tokens=4096,
    system = system,
    messages=[
        {
            "role": "user",
            "content": f"<query>{query}</query>",
        }
    ],
    model="claude-3-haiku-20240307",
    )
    return message.content[0].text


def process_input(input_text):
    processed_text = input_text.replace('\n', ' ').replace('\t', ' ')
    processed_text = re.sub(r'\s+', ' ', processed_text)
    processed_text = processed_text.strip()
    
    return processed_text

def generate_context(query):
    """Generate context based on a query."""
    hyde_query = groq_hyde(query)

    method_docs = method_table.search(hyde_query).limit(5).to_pandas()
    class_docs = class_table.search(hyde_query).limit(5).to_pandas()

    

    # no reranking first time because using 5 docs anyways

    # method_results = co.rerank(query=hyde_query, documents=method_docs['code'].tolist(), top_n=5, model='rerank-english-v3.0')
    # class_results = co.rerank(query=hyde_query, documents=class_docs['class_info'].tolist(), top_n=5, model='rerank-english-v3.0')


    temp_context = '\n'.join(method_docs['code'] + '\n'.join(class_docs['class_info']))

    # can switch to 70b for this if can reduce num of tokens
    hyde_query_v2 = anthropic_references(query, temp_context)

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
    classes_combined = "\n\n".join(f"File: {row['file_path']}\nClass Info:\n{row['class_info']}" for index, row in top_3_classes.iterrows())

    print("Context generation is complete.")

    return methods_combined + "\n\n" + classes_combined

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


            # Removing below code for references since fuzzy matching not working good enough
            """
            # get a better query for reference matching
            query_for_references = groq_query_for_references(query, context[:10000])
            # can directly dump this even though it has context and query

            logging.info("<query for references>")
            logging.info(query_for_references)
            logging.info("</query for references>")

            results_list = search_and_fetch_lines(query_for_references, codebase_path, 100, ix)

            logging.info(f"results list: {results_list}")
            code_list = []
            idx = 0
            for results in results_list:
                code_list.append( f"file_path: {results['absolute_path']}" + '\n'.join(results['lines']) )
                logging.info(f"{idx}  {results['lines']}")
                idx += 1

            references = '\n'.join(code_list)
            logging.info(f"length of references {len(references)}")
            
            context = anthropic_references(groq_hyde_v2, references) + context
            """

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
            
        
        response = anthropic_chat(query, context[:10000]) # token rate limit is problematic

        combined_response = f"Query: {query}\nResponse: {response}"

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
