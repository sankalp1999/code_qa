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
# Initialize Flask and other services
app = Flask(__name__)

app.secret_key = os.urandom(24)

# Redis connection
redis_client = redis.Redis(host='localhost', port=6379, db=0)
 

# Capture command line arguments for codebase path and language
if len(sys.argv) != 3:
    print("Usage: python app.py <language> <codebase_path>")
    sys.exit(1)

language = sys.argv[1]
codebase_path = sys.argv[2]

ix = open_existing_index(codebase_path)

normalized_path = os.path.normpath(os.path.abspath(codebase_path))
    
    # Extract the base name of the directory
codebase_folder_name = os.path.basename(normalized_path)

# Lancedb connection
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
                Predict the code for the query that might answer the query provided in input. The context is usually technical. 
                Just give the code based on the query, no additional text. Think step by step. Try to be concise.
                If the question is a general one, then try to include name of relevant docs like README.md or config files that may contain info.'''
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
                "content": f'''You are an software engineer specializing in programming language: {language}. 
                We need to improve and expand an original query: {query} with help of the context: {temp_context}. The new query should be like prediction of answer to 
                the query. Your task is to frame a better query using the context that might have more keywords in terms of code and method names with the code. 
                - If the query is code-related, provide code snippets with specific method names and keywords.
                - If query involves general questions and not specifically code related stuff, think and mention relevant files like refer README.md
                - If the query is about a specific method, try predicting how that method or function may look like in code or what libraries one may use focusing on latest ones.
                - try to keep query descriptive yet concise.
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
                1. frame a better, descriptive query under 4 lines with help of provided context focusing on keywords. Using your reasoning, point out what extra info / names / keywords are required for example when user asks questions about reposityr, you may mention README.md. You may add answer keywords based on your own knowledge to the query. Think on these lines.
                Output format: 
                2. Give a detailed summary from the context that might help answer the query.

                <output>
                <context> $context </context>
                <query> $query </query>

                '''
            },
            {
                "role": "user",
                "content": f"<query> {query} </query> <context> {context} </context>",
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
                "content": f"You are a software engineer. Using your knowledge and given the following {context}, explain user's queries. Highlight particular code blocks, method names, class names."
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
                1. with help of references provided, grab relevant info like documentation, code snippet, method name, class name that look relevant to answer the query
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
            "content": f"Here is my query:{query}",
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

    print("--queryv1--")
    print(method_docs['code'].tolist())
    print(class_docs['class_info'].tolist())
    print("-------")

    # no reranking first time because using 5 docs anyways

    # method_results = co.rerank(query=hyde_query, documents=method_docs['code'].tolist(), top_n=5, model='rerank-english-v3.0')
    # class_results = co.rerank(query=hyde_query, documents=class_docs['class_info'].tolist(), top_n=5, model='rerank-english-v3.0')


    temp_context = '\n'.join(method_docs['code'] + '\n'.join(class_docs['class_info']))

    # can switch to 70b for this if can reduce num of tokens
    hyde_query_v2 = anthropic_references(query, temp_context)

    print("-query_v2-")
    print(hyde_query_v2)
    print("---")

    method_docs = method_table.search(hyde_query_v2).limit(5).to_pandas()
    class_docs = class_table.search(hyde_query_v2).limit(5).to_pandas()

    print("---v2---")
    print(method_docs['code'].tolist())
    print(class_docs['class_info'].tolist())
    print("-------")

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


            print("-----context------", print(len(context)))
            print(len(process_input(context)))
            print(context)
            print("-----context_end-----")


            # get a better query for reference matching
            query_for_references = groq_query_for_references(query, context[:10000])
            # can directly dump this even though it has context and query

            print("<query for references>")
            print(query_for_references)
            print("</query for references>")

            results_list = search_and_fetch_lines(query_for_references, codebase_path, 100, ix)

            code_list = []
            for results in results_list:
                code_list.append( f"file_path: {results['absolute_path']}" + '\n'.join(results['lines']) )
            references = '\n'.join(code_list)

            print("reference length", len(references))

            # commenting below because context can get >10k characters because class_info can get upto 10k characters since it 
            # contains class code so need to potentially navigate around this
            # one possible solution is to get top 5 class names. write a function call that allows llm to retrieve 
            # class code on demand. this way, i won't have to load all classes in context at same time

            # context = context + anthropic_references(query_for_references, context, references)
            
            # not passing context now, earlier i wanted llm to decide what info to extract itself from the context
            # but avoiding that now
            context = anthropic_references(groq_hyde_v2, references) + context
   

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
