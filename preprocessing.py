import os
import sys
from treesitter import Treesitter, Language
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
WHITELIST_FILES = [".java",".py", ".js", ".rs", ".md"]
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
    "c": {
        "class": "struct_specifier",
        "method": "function_definition"
    }
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
        "child_field_name": "function"  # Placeholder, fill in if known
    },
    "rust": {
        "class": "identifier",
        "method": "call_expression",
        "child_field_name": "function"  # Placeholder, fill in if known
    },
    "c": {
        "class": "type_identifier",
        "method": "call_expression",
        "child_field_name": "function"  # Placeholder, fill in if known
    }
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
            identifier = node.child_by_field_name(child_field_name) # java:name, python:function
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
                name = node.child_by_field_name("name").text.decode() 
                if method == False:
                    print(f"Class Name: {name}")
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

    # for class_info in class_infos:
    #     print(f"class name:{class_info.name}, references:{class_info.references}")
    # for method_info in method_infos:
    #     print(f"method name:{method_info.name}, references:{method_info.references}")

    class_infos_dict = {class_info.name: class_info for class_info in class_infos}
    method_infos_dict = {method_info.method_key : method_info for method_info in method_infos}

    return class_infos_dict, method_infos_dict
    

def parse_code_files(file_list, codebase_language):
    class_data = []
    method_data = []

    
    class_infos_dict, method_infos_dict = get_references(file_list, codebase_language.value)

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
                references = []
            
            # Get the references for the current class from the class_infos_dict
                if class_name in class_infos_dict:
                    class_info = class_infos_dict[class_name]
                    references = class_info.references
                    # print("class_info got hit", class_info.references)

                class_data.append({
                    "file_path": code_file,
                    "class_name": class_name,
                    "constructor_declaration": class_node.constructor_declaration,
                    "method_declarations": "\n-----\n".join(class_node.method_declarations) if class_node.method_declarations else "",
                    "source_code": class_node.source_code,
                    "references": references # Initialize an empty set for references
                })

            for method_node in method_nodes:
                name = method_node.name
                references = []
                
                # Get the references for the current method from the method_infos_dict
                if name in method_infos_dict:
                    method_info = method_infos_dict[name]
                    # print("method_info got hit", method_info.references)
                    references = method_info.references

                method_data.append({
                    "file_path": code_file,
                    "class_name": "",  # Placeholder for class name
                    "name": method_node.name,
                    "doc_comment": method_node.doc_comment,
                    "source_code": method_node.method_source_code,
                    "references": references
                })
                
    return class_data, method_data


def extract_classes(content):
    pattern = re.compile(r'\b(class|interface|enum)\s+(\w+)')
    return {match.group(2) for match in pattern.finditer(content)}


def create_output_directory(codebase_path):
    # Normalize and get the absolute path
    normalized_path = os.path.normpath(os.path.abspath(codebase_path))
    
    # Extract the base name of the directory
    codebase_folder_name = os.path.basename(normalized_path)
    
    
    # Create the output directory under 'processed'
    output_directory = os.path.join("processed", codebase_folder_name)
    os.makedirs(output_directory, exist_ok=True)
    
    return output_directory


def write_class_data_to_csv(class_data, output_directory):
    output_file = os.path.join(output_directory, "class_data.csv")
    fieldnames = ["file_path", "class_name", "constructor_declaration", "method_declarations", "source_code", "references"]
    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        idx = 0
        for row in class_data:
            row["references"] = ",".join(map(str, row["references"]))
            writer.writerow(row)
            idx += 1
            print(f"row {idx} written")
            print(row)
            print("------------------")
    print(f"Class data written to {output_file}")

def write_method_data_to_csv(method_data, output_directory):
    output_file = os.path.join(output_directory, "method_data.csv")
    fieldnames = ["file_path", "class_name", "name", "doc_comment", "source_code", "references"]
    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in method_data:
            # Convert references list to a comma-separated string
            row["references"] = ",".join(map(str, row["references"]))
            writer.writerow(row)
    print(f"Method data written to {output_file}")

# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Please provide the codebase path and language as arguments.")
        sys.exit(1)
    codebase_language = sys.argv[1]
    codebase_path = sys.argv[2]
    

    # removing fuzzy search because not giving enough ROI
    """
    create_or_recreate_index(codebase_path)
    index_codebase(codebase_path)
    """
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

    output_directory = create_output_directory(codebase_path)
    # Write class data to CSV
    write_class_data_to_csv(class_data, output_directory)

    # Write method data to CSV
    write_method_data_to_csv(method_data, output_directory)