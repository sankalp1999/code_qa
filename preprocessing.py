import os
import sys
from treesitter import Treesitter, LanguageEnum
from collections import defaultdict
import re
import csv
from typing import List, Dict
from tree_sitter import Node
from tree_sitter_languages import get_language, get_parser

class CodeInfo:
    def __init__(self, name: str, method_key: str, file_path: str, references: List[Dict[str, str]]):
        self.name = name
        self.method_key = method_key
        self.file_path = file_path
        self.references = references

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
    for root, _, files in os.walk(codebase_path):
        if any(blacklist in root for blacklist in BLACKLIST_DIR):
            continue
        for file in files:
            file_ext = os.path.splitext(file)[1]
            if file_ext in WHITELIST_FILES:
                if file not in BLACKLIST_FILES and file != "docker-compose.yml":
                    file_path = os.path.join(root, file)
                    language = get_language_from_extension(file_ext)
                    if language:
                        file_list.append((file_path, language))
                    else:
                        print(f"Unsupported file extension {file_ext} in file {file_path}. Skipping.")
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
WHITELIST_FILES = [".java", ".py", ".js", ".rs", ".md"]  # Add other extensions as needed
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
        "method": "function_declaration"
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

def find_class_references(file_path: str, class_ref_node_type, class_name: str, ast: Node) -> List[Dict[str, str]]:
    references = []
    tree = ast
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == class_ref_node_type and node.text.decode() == class_name:
            reference = {
                "file": file_path,
                "line": node.start_point[0],
                "column": node.start_point[1],
                "text": node.parent.text.decode()
            }
            references.append(reference)
        stack.extend(node.children)
    return references

def find_method_references(file_path: str, method_ref_node_type, method_name: str, ast: Node, child_field_name) -> List[Dict[str, str]]:
    references = []

    def traverse_node(node):
        if node.type == method_ref_node_type:
            identifier = node.child_by_field_name(child_field_name)
            if identifier and identifier.text.decode() == method_name:
                reference = {
                    "file": file_path,
                    "line": node.start_point[0],
                    "column": node.start_point[1],
                    "text": node.text.decode()
                }
                references.append(reference)
        for child in node.children:
            traverse_node(child)

    traverse_node(ast.root_node)
    return references

def extract_code(file_path: str, parser, node_type: str, method) -> List[CodeInfo]:
    with open(file_path, "r") as file:
        code = file.read()
        tree = parser.parse(bytes(code, "utf8"))
        infos = []
        
        def traverse_node(node):
            if node.type == node_type:
                if method:
                    code_lines = node.text.decode().split("\n")
                    if code_lines:
                        method_key = "\n".join(code_lines[:2]) 
                else:
                    method_key = node.text.decode()    
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = name_node.text.decode()
                    infos.append(CodeInfo(name, method_key, file_path, []))
                
            for child in node.children:
                traverse_node(child)
        
        traverse_node(tree.root_node)
        return infos

def process_codebase(file_list, language: str):
    parser = get_parser(language)
    class_infos = []
    method_infos = []
    file_asts = {}

    class_node_type = NODE_TYPES[language]["class"]
    method_node_type = NODE_TYPES[language]["method"]

    class_ref_node_type = REFERENCE_IDENTIFIERS[language]["class"]
    method_ref_node_type = REFERENCE_IDENTIFIERS[language]["method"]
    child_field_name = REFERENCE_IDENTIFIERS[language]["child_field_name"]

    for file_path in file_list:
        if file_path not in file_asts:
            class_infos_in_file = extract_code(file_path, parser, class_node_type, method=False)
            class_infos.extend(class_infos_in_file)
            file_asts[file_path] = parser.parse(bytes(open(file_path, "r").read(), "utf8"))

    for class_info in class_infos:
        references = []
        for ref_file_path, ast in file_asts.items():
            if ref_file_path != class_info.file_path:
                references.extend(find_class_references(ref_file_path, class_ref_node_type, class_info.name, ast))
        class_info.references = references

    for file_path in file_list:
        method_infos_in_file = extract_code(file_path, parser, method_node_type, method=True)
        method_infos.extend(method_infos_in_file)

    for method_info in method_infos:
        references = []
        for ref_file_path, ast in file_asts.items():
            if ref_file_path != method_info.file_path:
                references.extend(find_method_references(ref_file_path, method_ref_node_type, method_info.name, ast, child_field_name))
        method_info.references = references

    return class_infos, method_infos

def get_references(file_list, codebase_language):
    class_infos, method_infos = process_codebase(file_list, codebase_language)

    class_infos_dict = {class_info.name: class_info for class_info in class_infos}
    method_infos_dict = {method_info.name: method_info for method_info in method_infos}

    return class_infos_dict, method_infos_dict

def parse_code_files(file_list):
    class_data = []
    method_data = []

    # Group files by language
    files_by_language = defaultdict(list)
    for file_path, language in file_list:
        files_by_language[language].append(file_path)

    # Dictionaries to collect class and method infos
    class_infos_dict = {}
    method_infos_dict = {}

    for language, files in files_by_language.items():
        # Process files for this language
        class_infos, method_infos = get_references(files, language.value)

        # Build dictionaries with keys including language
        class_infos_dict.update({(language, class_info.name): class_info for class_info in class_infos})
        method_infos_dict.update({(language, method_info.name): method_info for method_info in method_infos})

    for file_path, language in file_list:
        with open(file_path, "r", encoding="utf-8") as file:
            file_bytes = file.read().encode()
            treesitter_parser = Treesitter.create_treesitter(language)
            class_nodes, method_nodes = treesitter_parser.parse(file_bytes)

            for class_node in class_nodes:
                class_name = class_node.name
                references = []

                # Get the references for the current class from the class_infos_dict
                class_info_key = (language, class_name)
                if class_info_key in class_infos_dict:
                    class_info = class_infos_dict[class_info_key]
                    references = class_info.references

                class_data.append({
                    "file_path": file_path,
                    "class_name": class_name,
                    "constructor_declaration": "",
                    "method_declarations": "\n-----\n".join(class_node.method_declarations) if class_node.method_declarations else "",
                    "source_code": class_node.source_code,
                    "references": references
                })

            for method_node in method_nodes:
                name = method_node.name
                references = []

                # Get the references for the current method from the method_infos_dict
                method_info_key = (language, name)
                if method_info_key in method_infos_dict:
                    method_info = method_infos_dict[method_info_key]
                    references = method_info.references

                method_data.append({
                    "file_path": file_path,
                    "class_name": "",  # Placeholder for class name
                    "name": method_node.name,
                    "doc_comment": method_node.doc_comment,
                    "source_code": method_node.method_source_code,
                    "references": references
                })

    return class_data, method_data

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
            row["references"] = ",".join(map(str, row["references"]))
            writer.writerow(row)
    print(f"Class data written to {output_file}")

def write_method_data_to_csv(method_data, output_directory):
    output_file = os.path.join(output_directory, "method_data.csv")
    fieldnames = ["file_path", "class_name", "name", "doc_comment", "source_code", "references"]
    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in method_data:
            row["references"] = ",".join(map(str, row["references"]))
            writer.writerow(row)
    print(f"Method data written to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the codebase path as an argument.")
        sys.exit(1)
    codebase_path = sys.argv[1]

    files = load_files(codebase_path)
    class_data, method_data = parse_code_files(files)

    output_directory = create_output_directory(codebase_path)
    write_class_data_to_csv(class_data, output_directory)
    write_method_data_to_csv(method_data, output_directory)
