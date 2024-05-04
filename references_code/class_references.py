import os
from typing import List, Dict
from tree_sitter import Node
from tree_sitter_languages import get_language, get_parser

class ClassInfo:
    def __init__(self, name: str, code: str, file_path: str, references: List[Dict[str, str]]):
        self.name = name
        self.code = code
        self.file_path = file_path
        self.references = references

def extract_class_code(file_path: str, parser, class_identifier="class_definition") -> List[ClassInfo]:
    with open(file_path, "r") as file:
        code = file.read()
        tree = parser.parse(bytes(code, "utf8"))
        class_infos = []
        
        def traverse_node(node):
            if node.type == class_identifier:
                class_name = node.child_by_field_name("name").text.decode()
                class_code = node.text.decode()
                class_infos.append(ClassInfo(class_name, class_code, file_path, []))
            
            for child in node.children:
                traverse_node(child)
        
        traverse_node(tree.root_node)
        return class_infos

def find_class_references(file_path: str, class_name: str, ast: Node, exclude_file: str = None) -> List[Dict[str, str]]:
    references = []
    if file_path == exclude_file:
        return references

    tree = ast
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "identifier" and node.text.decode() == class_name:
            reference = {
                "file": file_path,
                "line": node.start_point[0],
                "column": node.start_point[1],
                "text": node.parent.parent.text.decode()
            }
            references.append(reference)
        stack.extend(node.children)
    return references

def process_codebase(files: List[str], language: str):
    parser = get_parser(language)
    class_infos = []
    file_asts = {}

    for file_path in files:
        if file_path not in file_asts:
            class_infos_in_file = extract_class_code(file_path, parser)
            class_infos.extend(class_infos_in_file)
            file_asts[file_path] = parser.parse(bytes(open(file_path, "r").read(), "utf8"))

    for class_info in class_infos:
        references = []
        for ref_file_path, ast in file_asts.items():
            if ref_file_path != class_info.file_path:
                references.extend(find_class_references(ref_file_path, class_info.name, ast, exclude_file=class_info.file_path))
        class_info.references = references

    return class_infos

# Usage
# codebase_directory = "/Users/sankalp/Desktop/qa_groq/codebase"
# class_infos = process_codebase(codebase_directory)
# for class_info in class_infos:
#     print(f"Class: {class_info.name}")
#     print("References:", len(class_info.references))
#     for reference in class_info.references:
#         print(f" File: {reference['file']}")
#         print(f" Line: {reference['line']}, Column: {reference['column']}")
#         print(f" Text: {reference['text']}")
#     print()