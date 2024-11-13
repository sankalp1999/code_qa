from flask import Flask, render_template, request, session, jsonify
import os
import sys
import lancedb
from lancedb.rerankers import AnswerdotaiRerankers
import re
import redis
import uuid
import logging
import markdown
from openai import OpenAI
import json
from dotenv import load_dotenv
from redis import ConnectionPool
import time
from concurrent.futures import ThreadPoolExecutor
import openai

load_dotenv()

from prompts import (
    HYDE_SYSTEM_PROMPT,
    HYDE_V2_SYSTEM_PROMPT,
    CHAT_SYSTEM_PROMPT,
    RERANK_PROMPT
)

# Configuration
CONFIG = {
    'SECRET_KEY': os.urandom(24),
    'REDIS_HOST': 'localhost',
    'REDIS_PORT': 6379,
    'REDIS_DB': 0,
    'REDIS_POOL_SIZE': 10,  # Add pool size configuration
    'LOG_FILE': 'app.log',
    'LOG_FORMAT': '%(asctime)s - %(message)s',
    'LOG_DATE_FORMAT': '%d-%b-%y %H:%M:%S'
}

# Logging setup
def setup_logging(config):
    # Create a formatter
    formatter = logging.Formatter(
        config['LOG_FORMAT'],
        datefmt=config['LOG_DATE_FORMAT']
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(config['LOG_FILE'])
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Get the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Add both handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Database setup
def setup_database(codebase_path):
    normalized_path = os.path.normpath(os.path.abspath(codebase_path))
    codebase_folder_name = os.path.basename(normalized_path)

    # lancedb connection
    uri = "database"
    db = lancedb.connect(uri)

    method_table = db.open_table(codebase_folder_name + "_method")
    class_table = db.open_table(codebase_folder_name + "_class")

    return method_table, class_table

# Application setup
def setup_app():
    app = Flask(__name__)
    app.config.update(CONFIG)
    
    # Setup logging
    app.logger = setup_logging(app.config)
    
    # Redis connection pooling setup
    app.redis_pool = ConnectionPool(
        host=app.config['REDIS_HOST'],
        port=app.config['REDIS_PORT'],
        db=app.config['REDIS_DB'],
        max_connections=app.config['REDIS_POOL_SIZE']
    )
    
    # Create Redis client using the connection pool
    app.redis_client = redis.Redis(connection_pool=app.redis_pool)
    
    # Markdown filter
    @app.template_filter('markdown')
    def markdown_filter(text):
        return markdown.markdown(text, extensions=['fenced_code', 'tables'])
    
    return app

# Create the Flask app
app = setup_app()

# OpenAI client setup
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
client = openai.OpenAI(
    api_key=os.environ.get("SAMBANOVA_API_KEY"),
    base_url="https://api.sambanova.ai/v1",
)


# Initialize the reranker
reranker = AnswerdotaiRerankers(column="source_code")

# Replace groq_hyde function
def openai_hyde(query):
    chat_completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=512,
        messages=[
            {
                "role": "system",
                "content": HYDE_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"Help predict the answer to the query: {query}",
            }
        ]
    )
    app.logger.info(f"First HYDE response: {chat_completion.choices[0].message.content}")
    return chat_completion.choices[0].message.content

def openai_hyde_v2(query, temp_context, hyde_query):
    chat_completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        messages=[
            {
                "role": "system",
                "content": HYDE_V2_SYSTEM_PROMPT.format(query=query, temp_context=temp_context)
            },
            {
                "role": "user",
                "content": f"Predict the answer to the query: {hyde_query}",
            }
        ]
    )
    app.logger.info(f"Second HYDE response: {chat_completion.choices[0].message.content}")
    return chat_completion.choices[0].message.content


def openai_chat(query, context):
    start_time = time.time()
    
    chat_completion = client.chat.completions.create(
        model='Meta-Llama-3.1-70B-Instruct',
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
    
    chat_time = time.time() - start_time
    app.logger.info(f"Chat response took: {chat_time:.2f} seconds")
    
    return chat_completion.choices[0].message.content

def rerank_using_small_model(query, context):
    start_time = time.time()
    
    chat_completion = client.chat.completions.create(
        model='Meta-Llama-3.1-8B-Instruct',
        messages=[
            {
                "role": "system",
                "content": RERANK_PROMPT.format(context=context)
            },
            {
                "role": "user",
                "content": query,
            }
        ]
    )
    
    chat_time = time.time() - start_time
    app.logger.info(f"Llama 8B reranker response took: {chat_time:.2f} seconds")
    
    return chat_completion.choices[0].message.content

def process_input(input_text):
    processed_text = input_text.replace('\n', ' ').replace('\t', ' ')
    processed_text = re.sub(r'\s+', ' ', processed_text)
    processed_text = processed_text.strip()
    
    return processed_text

def generate_context(query, rerank=False):
    start_time = time.time()
    
    # First HYDE call
    hyde_query = openai_hyde(query)
    hyde_time = time.time()
    app.logger.info(f"First HYDE call took: {hyde_time - start_time:.2f} seconds")

    # Concurrent execution of first database searches
    def search_method_table():
        return method_table.search(hyde_query).limit(5).to_pandas()

    def search_class_table():
        return class_table.search(hyde_query).limit(5).to_pandas()

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_method_docs = executor.submit(search_method_table)
        future_class_docs = executor.submit(search_class_table)
        method_docs = future_method_docs.result()
        class_docs = future_class_docs.result()

    first_search_time = time.time()
    app.logger.info(f"First DB search took: {first_search_time - hyde_time:.2f} seconds")

    temp_context = '\n'.join(method_docs['code'].tolist() + class_docs['source_code'].tolist())

    # Second HYDE call
    hyde_query_v2 = openai_hyde_v2(query, temp_context, hyde_query)
    second_hyde_time = time.time()
    app.logger.info(f"Second HYDE call took: {second_hyde_time - first_search_time:.2f} seconds")

    # Concurrent execution of second database searches
    def search_method_table_v2():
        return method_table.search(hyde_query_v2)

    def search_class_table_v2():
        return class_table.search(hyde_query_v2)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_method_search = executor.submit(search_method_table_v2)
        future_class_search = executor.submit(search_class_table_v2)
        method_search = future_method_search.result()
        class_search = future_class_search.result()

    search_time = time.time()
    app.logger.info(f"Second DB search took: {search_time - second_hyde_time:.2f} seconds")

    # Concurrent reranking if enabled
    app.logger.info(f"Reranking enabled: {rerank}")
    if rerank:
        rerank_start_time = time.time()  # Start timing before reranking
        
        def rerank_method_search():
            return method_search.rerank(reranker)

        def rerank_class_search():
            return class_search.rerank(reranker)

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_method_search = executor.submit(rerank_method_search)
            future_class_search = executor.submit(rerank_class_search)
            method_search = future_method_search.result()
            class_search = future_class_search.result()

        rerank_time = time.time()
        app.logger.info(f"Reranking took: {rerank_time - rerank_start_time:.2f} seconds")
    
    # Set final time reference point
    rerank_time = time.time() if rerank else search_time

    # Fetch top documents
    method_docs = method_search.limit(5).to_list()
    class_docs = class_search.limit(5).to_list()
    final_search_time = time.time()
    app.logger.info(f"Final DB search took: {final_search_time - rerank_time:.2f} seconds")

    # Combine documents
    top_3_methods = method_docs[:3]
    methods_combined = "\n\n".join(
        f"File: {doc['file_path']}\nCode:\n{doc['code']}" for doc in top_3_methods
    )

    top_3_classes = class_docs[:3]
    classes_combined = "\n\n".join(
        f"File: {doc['file_path']}\nClass Info:\n{doc['source_code']} References: \n{doc['references']}  \n END OF ROW {i}"
        for i, doc in enumerate(top_3_classes)
    )

    final_context = rerank_using_small_model(query, classes_combined + "\n" + methods_combined)

    app.logger.info("Context generation complete.")

    total_time = time.time() - start_time
    app.logger.info(f"Total context generation took: {total_time:.2f} seconds")
    return final_context

    
    # return methods_combined + "\n below is class or constructor related code \n" + classes_combined

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            # This is an AJAX request
            data = request.get_json()
            query = data['query']
            rerank = data.get('rerank', False)  # Extract rerank value
            user_id = session.get('user_id')
            if user_id is None:
                user_id = str(uuid.uuid4())
                session['user_id'] = user_id

            # Ensure rerank is a boolean
            rerank = True if rerank in [True, 'true', 'True', '1'] else False

            if '@codebase' in query:
                query = query.replace('@codebase', '').strip()
                context = generate_context(query, rerank)
                app.logger.info("Generated context for query with @codebase.")
                app.redis_client.set(f"user:{user_id}:chat_context", context)
            else:
                context = app.redis_client.get(f"user:{user_id}:chat_context")
                if context is None:
                    context = ""
                else:
                    context = context.decode()

            # Now, apply reranking during the chat response if needed
            response = openai_chat(query, context[:12000])  # Adjust as needed

            # Store the conversation history
            redis_key = f"user:{user_id}:responses"
            combined_response = {'query': query, 'response': response}
            app.redis_client.rpush(redis_key, json.dumps(combined_response))

            # Return the bot's response as JSON
            return jsonify({'response': response})

    # For GET requests and non-AJAX POST requests, render the template as before
    # Retrieve the conversation history to display
    user_id = session.get('user_id')
    if user_id:
        redis_key = f"user:{user_id}:responses"
        responses = app.redis_client.lrange(redis_key, -5, -1)
        responses = [json.loads(resp.decode()) for resp in responses]
        results = {'responses': responses}
    else:
        results = None

    return render_template('query_form.html', results=results)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python app.py <codebase_path>")
        sys.exit(1)

    codebase_path = sys.argv[1]
    
    # Setup database
    method_table, class_table = setup_database(codebase_path)
    
    app.logger.info("Server starting up...")  # Test log message
    app.run(host='0.0.0.0', port=5001)


# Main latency here is because of Context + LLM processing so need faster LLM


# SambaNova halves the total effective time
# 13-Nov-24 03:33:14 - First HYDE call took: 2.20 seconds
# 13-Nov-24 03:33:15 - First DB search took: 1.44 seconds
# 13-Nov-24 03:33:20 - Second HYDE call took: 4.91 seconds
# 13-Nov-24 03:33:22 - Second DB search took: 1.53 seconds
# 13-Nov-24 03:33:22 - Reranking enabled: True
# 13-Nov-24 03:33:22 - Reranking took: 0.00 seconds
# 13-Nov-24 03:33:22 - Final DB search took: 0.55 seconds
# 13-Nov-24 03:33:22 - Context generation complete.
# 13-Nov-24 03:33:22 - Total context generation took: 10.63 seconds
# 13-Nov-24 03:33:22 - Generated context for query with @codebase.
# 13-Nov-24 03:33:28 - Chat response took: 5.59 seconds

# 127.0.0.1 - - [13/Nov/2024 02:45:06] "GET / HTTP/1.1" 200 -
# 13-Nov-24 02:45:21 - First HYDE call took: 3.05 seconds
# 13-Nov-24 02:45:23 - First DB search took: 2.36 seconds
# 13-Nov-24 02:45:34 - Second HYDE call took: 10.82 seconds
# 13-Nov-24 02:45:36 - Reranking took: 2.44 seconds
# 13-Nov-24 02:45:37 - Second DB search took: 0.65 seconds
# 13-Nov-24 02:45:37 - Context generation complete.
# 13-Nov-24 02:45:37 - Total context generation took: 19.32 seconds
# 13-Nov-24 02:45:37 - Generated context for query with @codebase.
# 13-Nov-24 02:46:00 - Chat response took: 23.01 seconds


# 127.0.0.1 - - [13/Nov/2024 03:01:37] "POST / HTTP/1.1" 200 -
# 13-Nov-24 03:01:54 - First HYDE call took: 3.18 seconds
# 13-Nov-24 03:01:55 - First DB search took: 1.28 seconds
# 13-Nov-24 03:02:02 - Second HYDE call took: 6.87 seconds
# 13-Nov-24 03:02:03 - Second DB search took: 0.85 seconds
# 13-Nov-24 03:02:03 - Reranking took: 0.00 seconds
# 13-Nov-24 03:02:04 - Final DB search took: 0.68 seconds
# 13-Nov-24 03:02:04 - Context generation complete.
# 13-Nov-24 03:02:04 - Total context generation took: 12.86 seconds
# 13-Nov-24 03:02:04 - Generated context for query with @codebase.
# 13-Nov-24 03:02:26 - Chat response took: 22.19 seconds

