# CodeQA


This project is a code search and query system that allows you to index and search through a codebase using natural language queries. It supports python, rust, javascript and java. It provides a minimal ui for easy interaction.

Note for users: If you have a new account for openai or anthropic, you may 
get hit with tokens per minute rate limits. 

TODO:
- improve latency - shift to groq when they raise rate limits
- better name to be decided


## Features

- Index and search through codebases in various programming languages (Java, Python, Rust, JavaScript)
- Natural language query interface for code search
- Context-aware search results with relevant code snippets
- Chat-based interaction for code-related queries

## Prerequisites

- Python 3.6 or higher
- Redis server running on `localhost:6379`

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/sankalp1999/code_qa.git
   ```

2. Navigate to the project directory:

   ```bash
   cd code_qa
   ```

3. Set up a Python virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the redis server
```
redis-server
```

## Configuration

Set the environment variables for the API keys:

```bash
export OPENAI_KEY="your-openai-api-key"
export ANTHROPIC_KEY="your-anthropic-api-key"
export GROQ_KEY="your-groq-api-key"
export COHERE_KEY="your-cohere-api-key"
```

## Building the Codebase Index

To build the index for the codebase, run the following script:


```
chmod +x index_codebase.sh
```

```bash
./index_codebase.sh
```

Follow the prompts to enter the language and folder path of your codebase.

## Usage

To start the server, use the following command:

language commands should look like: `javascript, python, rust, java`

```bash
python app.py <language> <folder_path>
```

For example, to analyze a JavaScript project located in `/Users/sankalp/Documents/code2prompt/twitter-circle`, run:

```bash
python app.py javascript /Users/sankalp/Documents/code2prompt/twitter-circle
```

Once the server is running, open a web browser and navigate to `http://localhost:5000` to access the code search and query interface.

## Supported Languages

- Java
- Python
- Rust
- JavaScript

## Technologies Used

- Flask: server and UI
- Treesitter: parsing methods, classes, constructor declarations in a language agnostic way using the abstract syntax tree
- Whoosh: full-text search library for indexing and querying code
- LanceDB: vector db for storing and searching code embeddings
- Redis: in-memory data store for caching and session management
- OpenAI, Anthropic, Cohere, Groq: Openai, Anthropic, Groq for chat functionalities and Cohere for reranker

## How It Works

Stay tuned for an upcoming blog post that will provide detailed insights and advancements related to this project.

Summary

This project is essentially top-K RAG with bunch of other small hackery / tricks.

Code QA essentially boils down to a code search problem since you need to map from natural language to actual keywords. 

You need to provide relevant context to the LLM so that it can actually answer the questions and not hallucinate.

- I use tree-sitter ast (abstract syntax tree) to get all methods, classes, constructor declarations.
- different tree-sitter language implementations have slightly different syntax so accommodate for that
- generate short documentation for each of the methods -> to improve the embedding search, also query will look more like documentation
- Embed the methods and classes in separate vector db tables
- REMOVED -> generate a full-text based index using whoosh library. this is a language agnostic way to get references once the keywords are clear
- At retrieval time query -> Hyde query -> get context -> get better query using context -> get references -> plug new context into LLM 
- get answer


## License

This project is licensed under the [MIT License](LICENSE).
