import os
import sys
from treesitter import Treesitter, Language
from collections import defaultdict
import re
import csv
from fuzzy_search import create_or_recreate_index, index_codebase

def load_files(codebase_path):
    file_list = []
    for root, _, files in os.walk(codebase_path):
        if any(blacklist in root for blacklist in BLACKLIST_DIR):
            continue
        for file in files:
            file_ext = os.path.splitext(file)[1]
            if any(whitelist == file_ext for whitelist in WHITELIST_FILES):
                if file not in BLACKLIST_FILES and file != "docker-compose.yml":
                    file_path = os.path.join(root, file)
                    file_list.append(file_path)
    return file_list

BLACKLIST_DIR = [
    "__pycache__",
    ".pytest_cache",
    ".venv",
    ".git",
    ".idea",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    ".vscode",
    ".github",
    ".gitlab",
    ".angular",
    "cdk.out",
    ".aws-sam",
    ".terraform"
]
WHITELIST_FILES = [".java", ".sql", ".py", ".js", ".rs", ".md"]
BLACKLIST_FILES = ["docker-compose.yml"]

def parse_code_files(file_list, codebase_language):
    class_data = []
    method_data = []


    programming_language = codebase_language

    for code_file in file_list:
        with open(code_file, "r", encoding="utf-8") as file:
            file_bytes = file.read().encode()
            file_extension = os.path.splitext(code_file)[1]
            if programming_language == Language.UNKNOWN:
                print(f"Unknown programming language for file: {code_file}")
                continue
            treesitter_parser = Treesitter.create_treesitter(programming_language)
            class_nodes, method_nodes = treesitter_parser.parse(file_bytes)
            
            for class_node in class_nodes:
                class_name = class_node.name
                print(class_name)
                
                class_data.append({
                    "file_path": code_file,
                    "class_name": class_name,
                    "constructor_declaration": class_node.constructor_declaration,
                    "method_declarations": class_node.method_declarations,
                    "references": set()  # Initialize an empty set for references
                })
            
            for method_node in method_nodes:
                method_data.append({
                    "file_path": code_file,
                    "class_name": "",  # Placeholder for class name
                    "name": method_node.name,
                    "doc_comment": method_node.doc_comment,
                    "source_code": method_node.method_source_code
                })
                
    return class_data, method_data

def find_class_references(file_list, class_data):
    class_to_file = defaultdict(set)

    for code_file in file_list:
        with open(code_file, "r", encoding="utf-8") as file:
            content = file.read()
            classes = extract_classes(content)
            for cls in classes:
                class_to_file[cls].add(os.path.basename(code_file))

    for class_info in class_data:
        code_file = class_info["file_path"]
        with open(code_file, "r", encoding="utf-8") as file:
            content = file.read()
            for classname, origin_files in class_to_file.items():
                if classname in content:
                    for origin_file in origin_files:
                        if origin_file != os.path.basename(code_file):
                            class_info["references"].add(origin_file)

def extract_classes(content):
    pattern = re.compile(r'\b(class|interface|enum)\s+(\w+)')
    return {match.group(2) for match in pattern.finditer(content)}


def create_output_directory(codebase_path):
    # Normalize and get the absolute path
    normalized_path = os.path.normpath(os.path.abspath(codebase_path))
    
    # Extract the base name of the directory
    codebase_folder_name = os.path.basename(normalized_path)
    
    print("codebase_folder_name:", codebase_folder_name)
    
    # Create the output directory under 'processed'
    output_directory = os.path.join("processed", codebase_folder_name)
    os.makedirs(output_directory, exist_ok=True)
    
    return output_directory


def write_class_data_to_csv(class_data, output_directory):
    output_file = os.path.join(output_directory, "class_data.csv")
    fieldnames = ["file_path", "class_name", "constructor_declaration", "method_declarations", "references"]
    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for class_info in class_data:
            class_info["references"] = ",".join(class_info["references"])  # Convert set to string
            writer.writerow(class_info)
    print(f"Class data written to {output_file}")

def write_method_data_to_csv(method_data, output_directory):
    output_file = os.path.join(output_directory, "method_data.csv")
    fieldnames = ["file_path", "class_name", "name", "doc_comment", "source_code"]
    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(method_data)
    print(f"Method data written to {output_file}")

# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Please provide the codebase path and language as arguments.")
        sys.exit(1)
    codebase_language = sys.argv[1]
    codebase_path = sys.argv[2]
    
    create_or_recreate_index(codebase_path)
    index_codebase(codebase_path)

    programming_language = Language.UNKNOWN 
    if codebase_language.lower() == "java":
        programming_language = Language.JAVA
    elif codebase_language.lower() == "python":
        programming_language = Language.PYTHON
    elif codebase_language.lower() == "javascript":
        programming_language = Language.JAVASCRIPT
    elif codebase_language.lower() == "rust":
        programming_language = Language.RUST

    files = load_files(codebase_path)
    class_data, method_data = parse_code_files(files, programming_language)

    if(programming_language == Language.JAVA):
        find_class_references(files, class_data)

    output_directory = create_output_directory(codebase_path)
    # Write class data to CSV
    write_class_data_to_csv(class_data, output_directory)

    # Write method data to CSV
    write_method_data_to_csv(method_data, output_directory)