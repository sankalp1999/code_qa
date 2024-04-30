import os
from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.qparser import QueryParser, FuzzyTermPlugin
import shutil

def create_or_recreate_index(directory):
    print("Creating or recreating index...")
    base_name = os.path.basename(directory.rstrip(os.sep))
    index_dir = os.path.join("processed", base_name, "index_files")
    print(f"Index directory: {index_dir}")

    schema = Schema(
        title=TEXT(stored=True),
        content=TEXT(stored=True),
        line=ID(stored=True),
        filepath=TEXT(stored=True)  # Store the filepath
    )

    # Remove existing index directory if it exists and recreate the index
    if os.path.exists(index_dir):
        print("Index directory exists. Removing and recreating...")
        shutil.rmtree(index_dir)
    os.makedirs(index_dir, exist_ok=True)
    ix = create_in(index_dir, schema)
    return ix

def open_existing_index(directory):
    print("Opening existing index...")
    base_name = os.path.basename(directory.rstrip(os.sep))
    index_dir = os.path.join("processed", base_name, "index_files")
    print(f"Index directory: {index_dir}")

    if os.path.exists(index_dir) and any(file.endswith(".toc") for file in os.listdir(index_dir)):
        print("Index files found. Opening index...")
        return open_dir(index_dir)
    else:
        print("No index found or index files are missing.")
        return None



def index_codebase(directory):
    print(f"Indexing files in directory {directory}")
    ix = open_existing_index(directory)
    writer = ix.writer()
    exclude_dirs = ['.git', 'node_modules', 'venv']  # Add more directories to exclude if needed
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]  # Exclude specified directories
        for file in files:
            filepath = os.path.join(root, file)
            print(f"Processing {filepath}")  # Debug: Check file paths being processed
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    for i, line in enumerate(f, 1):  # Start counting from line 1
                        line_content = line.strip()
                        if line_content:  # Only index non-empty lines
                            writer.add_document(title=file, content=line_content, line=str(i), filepath=filepath)
            except Exception as e:
                print(f"Error reading {file}: {e} - {filepath}")  # Include exception details
    writer.commit()
    print("Indexing completed.")


def search_and_fetch_lines(query_str, directory, line_range, ix):
    print(f"Searching for '{query_str}' in directory {directory}")
    # ix = open_existing_index(directory)
    results_list = []
    with ix.searcher() as searcher:
        parser = QueryParser("content", ix.schema)
        parser.add_plugin(FuzzyTermPlugin())
        query = parser.parse(f"{query_str}~")
        results = searcher.search(query, limit=5)
        if results:
            for result in results:
                file_path = result['filepath']  # Use the stored filepath
                absolute_path = os.path.abspath(file_path)  # Get the absolute file path
                start_line = max(1, int(result['line']) - line_range // 2)
                end_line = start_line + line_range
                lines = []
                with open(file_path, 'r') as file:
                    for i, line in enumerate(file, 1):
                        if i >= start_line and i <= end_line:
                            lines.append(f"{i}: {line.strip()}")
                result_dict = {
                    'file': result['title'],
                    'line': result['line'],
                    'content': result['content'],
                    'path': result['filepath'],
                    'absolute_path': absolute_path,  # Add the absolute file path to the result_dict
                    'lines': lines
                }
                results_list.append(result_dict)
    return results_list

if __name__ == "__main__":
    directory = "/Users/sankalp/Desktop/qa_groq/codebase"  # Replace with the actual path to your codebase
    create_or_recreate_index(directory)
    index_codebase(directory)
    search_and_fetch_lines("petcontroller", directory)  # Replace "your_search_term" with the term you are searching for
