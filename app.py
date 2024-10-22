from flask import Flask, render_template, request, session
import os
import sys
import lancedb
import re
import redis
import uuid
from loguru import logging
import markdown
from openai import OpenAI
from lancedb.rerankers import AnswerdotaiRerankers
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
    
    # Redis setup
    app.redis_client = redis.Redis(
        host=app.config['REDIS_HOST'],
        port=app.config['REDIS_PORT'],
        db=app.config['REDIS_DB']
    )
    
    # Markdown filter
    @app.template_filter('markdown')
    def markdown_filter(text):
        return markdown.markdown(text, extensions=['fenced_code', 'tables'])
    
    return app

# Create the Flask app
app = setup_app()

# OpenAI client setup
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# This is how it looks like in the source code, so we don't need to pass arguments
# def __init__(
#         self,
#         model_type="colbert",
#         model_name: str = "answerdotai/answerai-colbert-small-v1",
#         column: str = "text",
#         return_score="relevance",
#         **kwargs,
#     ):

reranker = AnswerdotaiRerankers()

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

# TODO double check this o
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

    hyde_query_v2 = openai_query_for_references(query, temp_context)

    logging.info("-query_v2-")
    logging.info(hyde_query_v2)

    method_docs = method_table.search(hyde_query_v2).rerank(reranker).limit(5).to_list()
    class_docs = class_table.search(hyde_query_v2).rerank(reranker).limit(5).to_list()

    top_3_methods = method_docs[:3]
    methods_combined = "\n\n".join(f"File: {doc['file_path']}\nCode:\n{doc['code']}" for doc in top_3_methods)

    top_3_classes = class_docs[:3]
    classes_combined = "\n\n".join(f"File: {doc['file_path']}\nClass Info:\n{doc['class_info']} References: \n{doc['references']}  \n END OF ROW {i}" for i, doc in enumerate(top_3_classes))


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
    results = None
    if request.method == 'POST':
        query = request.form['query']
        action = request.form['action']
        
        user_id = session.get('user_id')
        if user_id is None:
            user_id = str(uuid.uuid4())
            session['user_id'] = user_id

        if action == 'Context':
            # Context generation
            context = generate_context(query)
            app.logger.info("-----context------")
            app.logger.info(f"Context length: {len(process_input(context))}")
            app.logger.info(context)
            app.logger.info("-----context_end-----")

        elif action == 'Chat':
            # Retrieve context
            redis_key = f"user:{user_id}:chat_history"
            context = app.redis_client.get(redis_key)
            app.logger.info("INSIDE CHAT CONTEXT: %s", context)
            if context is None:
                context = ""
            else:
                app.logger.info("Found context")
                context = context.decode()
        
        response = openai_chat(query, context[:12000])  # token rate limit is problematic

        combined_response = f"Query: {query} \n\n Response: {response}"

        redis_key = f"user:{user_id}:responses"
        app.redis_client.rpush(redis_key, combined_response)

        # Update chat history in Redis
        new_chat_history = (context + f"\nQuery: {query}\nResponse: {response}").strip()
        app.redis_client.set(f"user:{user_id}:chat_history", new_chat_history)

        # Retrieve the last 3 responses for the current user from Redis
        responses = app.redis_client.lrange(redis_key, -3, -1)
        responses = [response.decode() for response in responses]

        results = {
            'response': response,
            'responses': responses
        }

    return render_template('query_form.html', results=results)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python app.py <language> <codebase_path>")
        sys.exit(1)

    language = sys.argv[1]
    codebase_path = sys.argv[2]
    
    # Setup database
    method_table, class_table = setup_database(codebase_path)
    
    app.run(debug=True)
