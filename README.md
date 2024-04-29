README WIP

Might post a blog on this. Went through many ideas but not implementing all.

Question answering on codebase using topK RAG and Treesitter AST. Supports python, rust, javascript, java

`python3 -m venv venv`

`source venv/bin/activate`

`pip install -r requirements.txt`

Set OPENAI_KEY, ANTHROPIC_KEY, GROQ_KEY, COHERE_KEY (Will reduce these soon)

To build codebase index -> `index.codebase.sh` 

To run server 

`python app.py <language> <folder_path>`

e.g 

python app.py javascript /Users/sankalp/Documents/code2prompt/twitter-circle   