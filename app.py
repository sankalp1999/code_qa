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

load_dotenv()

from prompts import (
    HYDE_SYSTEM_PROMPT,
    HYDE_V2_SYSTEM_PROMPT,
    REFERENCES_SYSTEM_PROMPT,
    CHAT_SYSTEM_PROMPT  
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
    logging.basicConfig(
        filename=config['LOG_FILE'],
        level=logging.INFO,
        format=config['LOG_FORMAT'],
        datefmt=config['LOG_DATE_FORMAT']
    )
    # Return a logger instance
    return logging.getLogger(__name__)

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
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# Initialize the reranker
reranker = AnswerdotaiRerankers(column="source_code")

# Replace groq_hyde function
def openai_hyde(query):
    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",
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
    return chat_completion.choices[0].message.content

def openai_hyde_v2(query, temp_context, hyde_query):
    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",
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

def generate_context(query, rerank=False):
    hyde_query = openai_hyde(query)

    method_docs = method_table.search(hyde_query).limit(5).to_pandas()
    class_docs = class_table.search(hyde_query).limit(5).to_pandas()

    temp_context = '\n'.join(method_docs['code'] + '\n'.join(class_docs['source_code']) )

    hyde_query_v2 = openai_query_for_references(query, temp_context)

    logging.info("-query_v2-")
    logging.info(hyde_query_v2)

    method_search = method_table.search(hyde_query_v2)
    class_search = class_table.search(hyde_query_v2)

    if rerank:
        method_search = method_search.rerank(reranker)
        class_search = class_search.rerank(reranker)

    method_docs = method_search.limit(5).to_list()
    class_docs = class_search.limit(5).to_list()

    top_3_methods = method_docs[:3]
    methods_combined = "\n\n".join(f"File: {doc['file_path']}\nCode:\n{doc['code']}" for doc in top_3_methods)

    top_3_classes = class_docs[:3]
    classes_combined = "\n\n".join(f"File: {doc['file_path']}\nClass Info:\n{doc['source_code']} References: \n{doc['references']}  \n END OF ROW {i}" for i, doc in enumerate(top_3_classes))

    app.logger.info("Classes Combined:")
    app.logger.info("-" * 40)
    app.logger.info(classes_combined)
    app.logger.info(f"Length of classes_combined: {len(classes_combined)}")
    app.logger.info("-" * 40)

    app.logger.info("Methods Combined:")
    app.logger.info("-" * 40)
    app.logger.info(methods_combined)
    app.logger.info("-" * 40)

    app.logger.info("Context generation complete.")

    return methods_combined + "\n below is class or constructor related code \n" + classes_combined

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
    
    app.run(host='0.0.0.0', port=5001)
