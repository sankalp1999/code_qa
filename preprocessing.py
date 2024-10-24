import os
import sys
from treesitter import Treesitter, LanguageEnum
from collections import defaultdict
import csv
from typing import List, Dict
from tree_sitter import Node
from tree_sitter_languages import get_language, get_parser

# Define your BLACKLIST_DIR, WHITELIST_FILES, NODE_TYPES, and REFERENCE_IDENTIFIERS here
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
WHITELIST_FILES = [".java", ".py", ".js", ".rs"]
BLACKLIST_FILES = ["docker-compose.yml"]

NODE_TYPES = {
    "python": {
        "class": "class_definition",
        "method": "function_definition"
    },
    "java": {
        "class": "class_declaration",
        "method": "method_declaration"
    },
    "rust": {
        "class": "struct_item",
        "method": "function_item"
    },
    "javascript": {
        "class": "class_declaration",
        "method": "method_definition"
    },
    # Add other languages as needed
}

REFERENCE_IDENTIFIERS = {
    "python": {
        "class": "identifier",
        "method": "call",
        "child_field_name": "function"
    },
    "java": {
        "class": "identifier",
        "method": "method_invocation",
        "child_field_name": "name"
    },
    "javascript": {
        "class": "identifier",
        "method": "call_expression",
        "child_field_name": "function"
    },
    "rust": {
        "class": "identifier",
        "method": "call_expression",
        "child_field_name": "function"
    },
    # Add other languages as needed
}

def get_language_from_extension(file_ext):
    FILE_EXTENSION_LANGUAGE_MAP = {
        ".java": LanguageEnum.JAVA,
        ".py": LanguageEnum.PYTHON,
        ".js": LanguageEnum.JAVASCRIPT,
        ".rs": LanguageEnum.RUST,
        # Add other extensions and languages as needed
    }
    return FILE_EXTENSION_LANGUAGE_MAP.get(file_ext)

def load_files(codebase_path):
    file_list = []
    for root, dirs, files in os.walk(codebase_path):
        dirs[:] = [d for d in dirs if d not in BLACKLIST_DIR]
        for file in files:
            file_ext = os.path.splitext(file)[1]
            if file_ext in WHITELIST_FILES:
                if file not in BLACKLIST_FILES:
                    file_path = os.path.join(root, file)
                    language = get_language_from_extension(file_ext)
                    if language:
                        file_list.append((file_path, language))
                    else:
                        print(f"Unsupported file extension {file_ext} in file {file_path}. Skipping.")
    return file_list

def find_class_references_in_ast(file_path, ast, class_name, language):
    references = []
    stack = [ast.root_node]
    while stack:
        node = stack.pop()
        if node.type == 'identifier' and node.text.decode() == class_name:
            # Check the context of the identifier to reduce false positives
            parent = node.parent
            if parent and parent.type in ['type', 'class_type', 'object_creation_expression']:
                reference = {
                    "file": file_path,
                    "line": node.start_point[0] + 1,
                    "column": node.start_point[1] + 1,
                    "text": parent.text.decode() if parent else node.text.decode()
                }
                references.append(reference)
        stack.extend(node.children)
    return references

def find_method_references_in_ast(file_path, ast, method_name, language):
    references = []
    stack = [ast.root_node]
    while stack:
        node = stack.pop()
        if node.type == 'identifier' and node.text.decode() == method_name:
            # Check if the identifier is part of a method call
            parent = node.parent
            if parent and parent.type in ['call_expression', 'method_invocation']:
                reference = {
                    "file": file_path,
                    "line": node.start_point[0] + 1,
                    "column": node.start_point[1] + 1,
                    "text": parent.text.decode()
                }
                references.append(reference)
        stack.extend(node.children)
    return references

def parse_code_files(file_list):
    class_data = []
    method_data = []

    all_class_names = set()
    all_method_names = set()

    files_by_language = defaultdict(list)
    for file_path, language in file_list:
        files_by_language[language].append(file_path)

    for language, files in files_by_language.items():
        treesitter_parser = Treesitter.create_treesitter(language)
        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as file:
                code = file.read()
                file_bytes = code.encode()
                class_nodes, method_nodes = treesitter_parser.parse(file_bytes)

                # Process class nodes
                for class_node in class_nodes:
                    class_name = class_node.name
                    all_class_names.add(class_name)
                    class_data.append({
                        "file_path": file_path,
                        "class_name": class_name,
                        "constructor_declaration": "",  # Extract if needed
                        "method_declarations": "\n-----\n".join(class_node.method_declarations) if class_node.method_declarations else "",
                        "source_code": class_node.source_code,
                        "references": []  # Will populate later
                    })

                # Process method nodes
                for method_node in method_nodes:
                    method_name = method_node.name
                    all_method_names.add(method_name)
                    method_data.append({
                        "file_path": file_path,
                        "class_name": method_node.class_name if method_node.class_name else "",
                        "name": method_name,
                        "doc_comment": method_node.doc_comment,
                        "source_code": method_node.method_source_code,
                        "references": []  # Will populate later
                    })

    return class_data, method_data, all_class_names, all_method_names

def find_references(file_list, class_names, method_names):
    references = {'class': defaultdict(list), 'method': defaultdict(list)}
    files_by_language = defaultdict(list)

    for file_path, language in file_list:
        files_by_language[language].append(file_path)

    for language, files in files_by_language.items():
        treesitter_parser = Treesitter.create_treesitter(language)
        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as file:
                code = file.read()
                file_bytes = code.encode()
                tree = treesitter_parser.parser.parse(file_bytes)
                ast = tree

                # Find class references
                for class_name in class_names:
                    class_refs = find_class_references_in_ast(file_path, ast, class_name, language.value)
                    references['class'][class_name].extend(class_refs)

                # Find method references
                for method_name in method_names:
                    method_refs = find_method_references_in_ast(file_path, ast, method_name, language.value)
                    references['method'][method_name].extend(method_refs)

    return references

def create_output_directory(codebase_path):
    normalized_path = os.path.normpath(os.path.abspath(codebase_path))
    codebase_folder_name = os.path.basename(normalized_path)
    output_directory = os.path.join("processed", codebase_folder_name)
    os.makedirs(output_directory, exist_ok=True)
    return output_directory

def write_class_data_to_csv(class_data, output_directory):
    output_file = os.path.join(output_directory, "class_data.csv")
    fieldnames = ["file_path", "class_name", "constructor_declaration", "method_declarations", "source_code", "references"]
    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in class_data:
            references = row.get("references", [])
            row["references"] = "; ".join([f"{ref['file']}:{ref['line']}:{ref['column']}" for ref in references])
            writer.writerow(row)
    print(f"Class data written to {output_file}")

def write_method_data_to_csv(method_data, output_directory):
    output_file = os.path.join(output_directory, "method_data.csv")
    fieldnames = ["file_path", "class_name", "name", "doc_comment", "source_code", "references"]
    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in method_data:
            references = row.get("references", [])
            row["references"] = "; ".join([f"{ref['file']}:{ref['line']}:{ref['column']}" for ref in references])
            writer.writerow(row)
    print(f"Method data written to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the codebase path as an argument.")
        sys.exit(1)
    codebase_path = sys.argv[1]

    files = load_files(codebase_path)
    class_data, method_data, class_names, method_names = parse_code_files(files)

    # Find references
    references = find_references(files, class_names, method_names)

    # Map references back to class and method data
    class_data_dict = {cd['class_name']: cd for cd in class_data}
    method_data_dict = {(md['class_name'], md['name']): md for md in method_data}

    for class_name, refs in references['class'].items():
        if class_name in class_data_dict:
            class_data_dict[class_name]['references'] = refs

    for method_name, refs in references['method'].items():
        # Find all methods with this name (since methods might have the same name in different classes)
        for key in method_data_dict:
            if key[1] == method_name:
                method_data_dict[key]['references'] = refs

    # Convert dictionaries back to lists
    class_data = list(class_data_dict.values())
    method_data = list(method_data_dict.values())

    output_directory = create_output_directory(codebase_path)
    write_class_data_to_csv(class_data, output_directory)
    write_method_data_to_csv(method_data, output_directory)
