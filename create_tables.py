import os
import sys
import pandas as pd
import lancedb
from lancedb.embeddings import EmbeddingFunctionRegistry
from lancedb.pydantic import LanceModel, Vector
import tiktoken
from dotenv import load_dotenv

load_dotenv()

def get_name_and_input_dir(codebase_path):
    # Normalize and get the absolute path
    normalized_path = os.path.normpath(os.path.abspath(codebase_path))
    
    # Extract the base name of the directory
    codebase_folder_name = os.path.basename(normalized_path)
    
    print("codebase_folder_name:", codebase_folder_name)
    
    # Create the output directory under 'processed'
    output_directory = os.path.join("processed", codebase_folder_name)
    os.makedirs(output_directory, exist_ok=True)
    
    return codebase_folder_name, output_directory

def get_special_files(directory):
    md_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.md', '.sh')):
                full_path = os.path.join(root, file)
                md_files.append(full_path)
    return md_files

def process_special_files(md_files):
    contents = {}
    for file_path in md_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            contents[file_path] = file.read()  # Store the content against the file path
    return contents


def create_markdown_dataframe(markdown_contents):
    # Create a DataFrame from markdown_contents dictionary
    df = pd.DataFrame(list(markdown_contents.items()), columns=['file_path', 'source_code'])
    
    # Format the source_code with file information and apply clipping
    df['source_code'] = df.apply(lambda row: f"File: {row['file_path']}\n\nContent:\n{clip_text_to_max_tokens(row['source_code'], MAX_TOKENS)}\n\n", axis=1)
    
    # Add placeholder "empty" for the other necessary columns
    for col in ['class_name', 'constructor_declaration', 'method_declarations', 'references']:
        df[col] = "empty"
    return df


# Check for environment variables and select embedding model
if os.getenv("JINA_API_KEY"):
    print("Using Jina")
    MODEL_NAME = "jina-embeddings-v3"
    registry = EmbeddingFunctionRegistry.get_instance()
    model = registry.get("jina").create(name=MODEL_NAME, max_retries=2)
    EMBEDDING_DIM = 1024  # Jina's dimension
    MAX_TOKENS = 4000   # Jina uses a different tokenizer so it's hard to predict the number of tokens
else:
    print("Using OpenAI")
    MODEL_NAME = "text-embedding-3-large"
    registry = EmbeddingFunctionRegistry.get_instance()
    model = registry.get("openai").create(name=MODEL_NAME, max_retries=2)
    EMBEDDING_DIM = model.ndims()  # OpenAI's dimension
    MAX_TOKENS = 8000    # OpenAI's token limit

class Method(LanceModel):
    code: str = model.SourceField()
    method_embeddings: Vector(EMBEDDING_DIM) = model.VectorField()
    file_path: str
    class_name: str
    name: str
    doc_comment: str
    source_code: str
    references: str

class Class(LanceModel):
    source_code: str = model.SourceField()
    class_embeddings: Vector(EMBEDDING_DIM) = model.VectorField()
    file_path: str
    class_name: str
    constructor_declaration: str
    method_declarations: str
    references: str

def clip_text_to_max_tokens(text, max_tokens, encoding_name='cl100k_base'):
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    original_token_count = len(tokens)
    
    print(f"\nOriginal text ({original_token_count} tokens):")
    print("=" * 50)
    print(text[:200] + "..." if len(text) > 200 else text)  # Print first 200 chars for preview
    
    if original_token_count > max_tokens:
        tokens = tokens[:max_tokens]
        clipped_text = encoding.decode(tokens)
        print(f"\nClipped text ({len(tokens)} tokens):")
        print("=" * 50)
        print(clipped_text[:200] + "..." if len(clipped_text) > 200 else clipped_text)
        return clipped_text
    
    return text

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <code_base_path>")
        sys.exit(1)

    codebase_path = sys.argv[1]

    table_name, input_directory = get_name_and_input_dir(codebase_path)
    method_data_file = os.path.join(input_directory, "method_data.csv")
    class_data_file = os.path.join(input_directory, "class_data.csv")

    special_files = get_special_files(codebase_path)
    special_contents = process_special_files(special_files)

    method_data = pd.read_csv(method_data_file)
    class_data = pd.read_csv(class_data_file)

    print(class_data.head())

    uri = "database"
    db = lancedb.connect(uri)

    # if codebase_path in db:
    #     print('exists already')
    #     table = db[codebase_path]
    # else:
    try:
        table = db.create_table(
            table_name + "_method", 
            schema=Method, 
            mode="overwrite",
            on_bad_vectors='drop'
        )

        method_data['code'] = method_data['source_code']
        null_rows = method_data.isnull().any(axis=1)

        if null_rows.any():
            print("Null values found in method_data. Replacing with 'empty'.")
            method_data = method_data.fillna('empty')
        else:
            print("No null values found in method_data.")

        # Add the concatenated data to the table
        print("Adding method data to table")
        table.add(method_data)
    
        class_table = db.create_table(
            table_name + "_class", 
            schema=Class, 
            mode="overwrite",
            on_bad_vectors='drop'
        )
        null_rows = class_data.isnull().any(axis=1)
        if null_rows.any():
            print("Null values found in class_data. Replacing with 'empty'.")
            class_data = class_data.fillna('')
        else:
            print("No null values found in class_data.")

        # row wise 
        class_data['source_code'] = class_data.apply(lambda row: "File: " + row['file_path'] + "\n\n" +
                                                        "Class: " + row['class_name'] + "\n\n" +
                                                        "Source Code:\n" + 
                                                        clip_text_to_max_tokens(row['source_code'], MAX_TOKENS) + "\n\n", axis=1)

        # TODO a misc content table is possible? where i dump other stuff like text files, markdown, config files, toml files etc.
        # print(markdown_contents)
        # class_table.add(markdown_contents) 
        # add after something because chance class_data may be empty
        if len(class_data) == 0:
            columns = ['source_code', 'file_path', 'class_name', 'constructor_declaration', 'method_declarations', 'references']
            empty_data = {col: ["empty"] for col in columns}

            class_data = pd.DataFrame(empty_data)
            
        print("Adding class data to table")
        class_table.add(class_data)

        if len(special_contents) > 0:
            markdown_df = create_markdown_dataframe(special_contents)
            print(f"Adding {len(markdown_df)} special files to table")
            class_table.add(markdown_df)

        print("Embedded method data successfully")
        print("Embedded class data successfully")

    except Exception as e:
        if codebase_path in db:
            db.drop_table(codebase_path)
        raise e
