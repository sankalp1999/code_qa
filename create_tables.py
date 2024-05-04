import os
import sys
import pandas as pd
import lancedb
from lancedb.embeddings import EmbeddingFunctionRegistry
from lancedb.pydantic import LanceModel, Vector

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

def get_markdown_files(directory):
    md_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                full_path = os.path.join(root, file)
                md_files.append(full_path)
    return md_files

def read_and_process_markdown_files(md_files):
    contents = {}
    for file_path in md_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            contents[file_path] = file.read()  # Store the content against the file path
    return contents


def create_markdown_dataframe(markdown_contents):
    # Create a DataFrame from markdown_contents dictionary
    df = pd.DataFrame(list(markdown_contents.items()), columns=['file_path', 'class_info'])
    # Add placeholder "empty" for the other necessary columns
    for col in ['class_name', 'constructor_declaration', 'method_declarations', 'references']:
        df[col] = "empty"
    return df


MODEL_NAME = "text-embedding-3-large"
registry = EmbeddingFunctionRegistry.get_instance()
model = registry.get("openai").create(name=MODEL_NAME, max_retries=2)

class Method(LanceModel):
    code: str = model.SourceField()
    method_embeddings: Vector(model.ndims()) = model.VectorField()
    file_path: str
    class_name: str
    name: str
    doc_comment: str
    source_code: str
    llm_comments: str
    references: str

class Class(LanceModel):
    class_info: str = model.SourceField()
    class_embeddings: Vector(model.ndims()) = model.VectorField()
    file_path: str
    class_name: str
    constructor_declaration: str
    method_declarations: str
    references: str

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <language> <code_base_path>")
        sys.exit(1)

    language = sys.argv[1]
    codebase_path = sys.argv[2]

    table_name, input_directory = get_name_and_input_dir(codebase_path)
    method_data_file = os.path.join(input_directory, "method_data.csv")
    class_data_file = os.path.join(input_directory, "class_data.csv")

    md_files = get_markdown_files(codebase_path)
    markdown_contents = read_and_process_markdown_files(md_files)

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
        table = db.create_table(table_name + "_method", schema=Method, mode="overwrite")

        # Concatenate llm_comments and source_code fields
        method_data['code'] = method_data['llm_comments'] + '\n\n' + method_data['source_code']

        # Check for null values in method_data
        null_rows = method_data.isnull().any(axis=1)
        if null_rows.any():
            print("Null values found in method_data. Replacing with 'empty'.")
            method_data = method_data.fillna('empty')
        else:
            print("No null values found in method_data.")

        # Add the concatenated data to the table
        table.add(method_data)
        

    
        class_table = db.create_table(table_name + "_class", schema=Class, mode="overwrite")
        null_rows = class_data.isnull().any(axis=1)
        if null_rows.any():
            print("Null values found in class_data. Replacing with 'empty'.")
            class_data = class_data.fillna('')
        else:
            print("No null values found in class_data.")

        class_data['class_info'] = (
            f"File: {class_data['file_path']}\n\n"
            f"Class: {class_data['class_name']}\n\n"
            f"Source Code:\n{class_data['source_code']}\n\n"
        )

        # TODO a misc content table is possible? where i dump other stuff like text files, markdown, config files, toml files etc.
        # print(markdown_contents)
        # class_table.add(markdown_contents) 
        # add after something because chance class_data may be empty
        if len(class_data) == 0:
            columns = ['class_info', 'file_path', 'class_name', 'constructor_declaration', 'method_declarations', 'references']
            empty_data = {col: ["empty"] for col in columns}

            class_data = pd.DataFrame(empty_data)
            
        class_table.add(class_data)
        class_table.add(create_markdown_dataframe(markdown_contents))

        print("Embedded method data successfully")

        print("Embedded class data successfully")

    except Exception as e:
        if codebase_path in db:
            db.drop_table(codebase_path)
        raise e