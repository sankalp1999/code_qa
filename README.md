

# Setup

1. Clone the repository:

   ```bash
   git clone <repository-url>
   ```

2. Navigate to the project directory:

   ```bash
   cd <repository-name>
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

### Configuration

Set the environment variables for the API keys:

```bash
export OPENAI_KEY="your-openai-api-key"
export ANTHROPIC_KEY="your-anthropic-api-key"
export GROQ_KEY="your-groq-api-key"
export COHERE_KEY="your-cohere-api-key"
```

### Building the Codebase Index

To build the index for the codebase, run the following script:

```bash
./index.codebase.sh
```

### Usage

To start the server, use the following command:

```bash
python app.py <language> <folder_path>
```

For example, to analyze a JavaScript project located in `/Users/sankalp/Documents/code2prompt/twitter-circle`, run:

```bash
python app.py javascript /Users/sankalp/Documents/code2prompt/twitter-circle
```

### Blog

Consider following the upcoming blog for detailed insights and advancements on this project.

---

