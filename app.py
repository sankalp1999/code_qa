from flask import Flask, render_template, request, session
import os
import sys
import lancedb
import cohere
import pandas as pd
from anthropic import Anthropic
from fuzzy_search import open_existing_index, index_codebase, search_and_fetch_lines
# Initialize Flask and other services
app = Flask(__name__)

app.secret_key = os.urandom(24)
 

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
                "content": f"You are a software engineer who specializes in the programming language: {language}. Your job is to predict the code for the query. The context is usually technical. Just give the code based on the query, no additional text. Think step by step."
            },
            {
                "role": "user",
                "content": f"Help predict the answer to the query: {query} in the programming language: {language}.",
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
                "content": f'''Given the query:{query} and context:{context} , 
                1. frame a better, descriptive query under 4 lines with help of provided context focusing on keywords. Using your reasoning, point out what extra info / names / keywords are required for example when user asks questions about reposityr, you may mention README.md. You may add answer keywords based on your own knowledge to the query. Think on these lines.
                Output format: 
                <query> new query here </query>
                '''
            },
            {
                "role": "user",
                "content": f"<query> {query} </query> <context> {context}",
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

def anthropic_references(query, context, references):
    system = f'''Given the query:{query} and context:{context} and references: {references}, 
                1. reply with what all info such as code snippet, class name, method name etc. should be added from the references
                2. frame a better query under 2 lines with proper names and keywords with help of context.
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
            "content": f"{query} <context> {context} </context> <references> {references} </references> ",
        }
    ],
    model="claude-3-haiku-20240307",
    )
    return message.content[0].text

def generate_context(query):
    """Generate context based on a query."""
    hyde_query = groq_hyde(query)
    method_docs = method_table.search(hyde_query).limit(10).to_pandas()
    class_docs = class_table.search(hyde_query).limit(10).to_pandas()

    method_results = co.rerank(query=hyde_query, documents=method_docs['code'].tolist(), top_n=5, model='rerank-english-v3.0')

    print(method_docs['code'].tolist())
    print(class_docs['class_info'].tolist())
    class_results = co.rerank(query=hyde_query, documents=class_docs['class_info'].tolist(), top_n=5, model='rerank-english-v3.0')

    
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
        
        if action == 'Context':
            ## Context generation
            context = generate_context(query)
            query_for_references = groq_query_for_references(query, context[:6000])
            results_list = search_and_fetch_lines(query_for_references, codebase_path, 100, ix)

            code_list = []
            for results in results_list:
                code_list.append( f"file_path: {results['absolute_path']}" + '\n'.join(results['lines']) )
            references = '\n'.join(code_list)

            context = context + anthropic_references(query_for_references, context, references)
   

        elif action == 'Chat':
            # Retrieve context from session
            context = session.get('chat_history', "")
            
        response = anthropic_chat(query, context)

        combined_response = f"Query: {query}\nResponse: {response}"
        # Update responses in session
        if 'responses' not in session:
            session['responses'] = []
        session['responses'].append(combined_response)

        # Keep only the last 3 responses
        session['responses'] = session['responses']

        # Update chat history in session, append the current query and response
        new_chat_history = (context + f"\nQuery: {query}\nResponse: {response}").strip()
        # Ensure chat history does not exceed 5000 characters, keeping the end portion
        # session['chat_history'] = new_chat_history[-8000:]
        session['chat_history'] = new_chat_history


        results = {
            'response': response,
            'responses': session['responses']
        }

    return render_template('query_form.html', results=results)

if __name__ == "__main__":
    app.run(debug=True)
