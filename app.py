from flask import Flask, render_template, request, session
import os
import sys
import lancedb
import cohere
import pandas as pd

# Initialize Flask and other services
app = Flask(__name__)

app.secret_key = os.urandom(24)
 

# Capture command line arguments for codebase path and language
if len(sys.argv) != 3:
    print("Usage: python app.py <language> <codebase_path>")
    sys.exit(1)

language = sys.argv[1]
codebase_path = sys.argv[2]


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
cohere_key = os.environ.get("COHERE_RERANKER_API_KEY")
co = cohere.Client(cohere_key)



def groq_hyde(query):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"You are a software engineer who specializes in {language}. Your job is to predict the code for the query. The context is usually technical. Just give the code based on the query, no additional text. Think step by step."
            },
            {
                "role": "user",
                "content": f"Help predict the answer to the {query} in the {language}.",
            }
        ],
        model="llama3-70b-8192",
    )
    return chat_completion.choices[0].message.content


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
   

        elif action == 'Chat':
            # Retrieve context from session
            context = session.get('chat_history', "")
            
        response = groq_chat(query, context)

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
        session['chat_history'] = new_chat_history[-8000:]

        results = {
            'response': response,
            'responses': session['responses']
        }

    return render_template('query_form.html', results=results)

if __name__ == "__main__":
    app.run(debug=True)
