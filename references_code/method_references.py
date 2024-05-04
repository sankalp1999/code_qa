import os
from typing import List, Dict
from tree_sitter import Node
from tree_sitter_languages import get_language, get_parser

class MethodInfo:
    def __init__(self, name: str, code: str, file_path: str, references: List[Dict[str, str]]):
        self.name = name
        self.code = code
        self.file_path = file_path
        self.references = references

def traverse_java_files(directory: str) -> List[str]:
    java_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".java"):
                java_files.append(os.path.join(root, file))
    return java_files

def extract_method_code(file_path: str, parser) -> List[MethodInfo]:
    with open(file_path, "r") as file:
        code = file.read()
        tree = parser.parse(bytes(code, "utf8"))
        method_infos = []

        def traverse_node(node):
            if node.type == "method_declaration":
                method_name = node.child_by_field_name("name").text.decode()
                method_code = node.text.decode()
                method_infos.append(MethodInfo(method_name, method_code, file_path, []))
            for child in node.children:
                traverse_node(child)

        traverse_node(tree.root_node)
        return method_infos

def find_method_references(file_path: str, method_name: str, ast: Node) -> List[Dict[str, str]]:
    references = []

    def traverse_node(node):
        if node.type == "method_invocation":
            identifier = node.child_by_field_name("name")
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

def process_codebase(directory: str):
    java_files = traverse_java_files(directory)
    language = get_language("java")
    parser = get_parser("java")
    method_infos = []
    file_asts = {}

    for file_path in java_files:
        if file_path not in file_asts:
            method_infos_in_file = extract_method_code(file_path, parser)
            method_infos.extend(method_infos_in_file)
            file_asts[file_path] = parser.parse(bytes(open(file_path, "r").read(), "utf8"))

    for method_info in method_infos:
        references = []
        for ref_file_path, ast in file_asts.items():
            if ref_file_path != method_info.file_path:
                references.extend(find_method_references(ref_file_path, method_info.name, ast))
        method_info.references = references

    return method_infos

# Usage
codebase_directory = "/Users/sankalp/Desktop/qa_groq/codebase"
method_infos = process_codebase(codebase_directory)

for method_info in method_infos:
    print(f"Method: {method_info.name}")
    print("References:", len(method_info.references))
    for reference in method_info.references:
        print(f" File: {reference['file']}")
        print(f" Line: {reference['line']}, Column: {reference['column']}")
        print(f" Text: {reference['text']}")
    print()