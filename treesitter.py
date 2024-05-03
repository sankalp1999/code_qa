from abc import ABC
import tree_sitter
from tree_sitter_languages import get_language, get_parser
from enum import Enum

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Language(Enum):
    JAVA = "java"
    PYTHON = "python"
    RUST = "rust"
    JAVASCRIPT = "javascript"
    UNKNOWN = "unknown"

class TreesitterMethodNode:
    def __init__(
        self,
        name: "str | bytes | None",
        doc_comment: "str | None",
        method_source_code: "str | None",
        node: tree_sitter.Node,
    ):
        self.name = name
        self.doc_comment = doc_comment
        self.method_source_code = method_source_code or node.text.decode()
        self.node = node

class TreesitterClassNode:
    def __init__(
        self,
        name: "str | bytes | None",
        # method_source_code: "str | None",
        method_declarations: "str | None",
        constructor_declaration: "str | None",
        node: tree_sitter.Node,
    ):
        self.name = name
        # self.method_source_code = method_source_code or node.text.decode()
        self.method_declarations = method_declarations
        self.constructor_declaration = constructor_declaration 
        self.node = node

class Treesitter(ABC):
    def __init__(
        self,
        language: Language,
        method_declaration_identifier: str,
        name_identifier: str,
        doc_comment_identifier: str,
        class_declaration_identifier,
        constructor_declaration_identifier: str
    ):
        self.parser = get_parser(language.value)
        self.language = get_language(language.value)

        self.method_declaration_identifier = method_declaration_identifier
        self.method_name_identifier = name_identifier
        self.doc_comment_identifier = doc_comment_identifier

        self.class_declaration_identifier = class_declaration_identifier
        self.constructor_declaration_identifier = constructor_declaration_identifier
        # methods are already covered in methods

    @staticmethod
    def create_treesitter(language: Language) -> "Treesitter":
        if language == Language.JAVA:
            from treesitter_implementations import TreesitterJava
            return TreesitterJava()
        elif language == Language.PYTHON:
            from treesitter_implementations import TreesitterPython
            return TreesitterPython()
        elif language == Language.RUST:
            from treesitter_implementations import TreesitterRust
            return TreesitterRust()
        elif language == Language.JAVASCRIPT:
            from treesitter_implementations import TreesitterJavaScript
            return TreesitterJavaScript()
        else:
            raise ValueError("Unsupported language")

    def parse(self, file_bytes: bytes) -> tuple[list[TreesitterClassNode], list[TreesitterMethodNode]]:
        self.tree = self.parser.parse(file_bytes)
        class_results = []
        method_results = []

        classes = self._query_classes(self.tree.root_node)
        logging.info(f"Found classes: {classes}") 
        for class_node in classes:
            class_name = self._query_class_name(class_node)
            constructor_declarations = self._query_constructor_declarations(class_node)
            method_declarations = self._query_method_declarations(class_node)
            class_results.append(TreesitterClassNode(class_name, method_declarations, constructor_declarations, class_node))

        methods = self._query_all_methods(self.tree.root_node)
        for method in methods:
            method_name = self._query_method_name(method["method"])
            doc_comment = method["doc_comment"]
            method_results.append(TreesitterMethodNode(method_name, doc_comment, None, method["method"]))

        return class_results, method_results

    def _query_classes(self, node: tree_sitter.Node):
        classes = []
        if node.type == self.class_declaration_identifier:
            classes.append(node)
        else:
            for child in node.children:
                classes.extend(self._query_classes(child))
        return classes

    def _query_class_name(self, node: tree_sitter.Node):
        if node.type == self.class_declaration_identifier:
            return node.text.decode()
        return None

    def _query_method_declarations(self, node: tree_sitter.Node):
        # need to separately take identifier and parameters from tree
        # so just fetch first line of method code. taking 2 to get the annotation as well
        method_declarations = []
        if node.type == self.method_declaration_identifier:
            code_lines = node.text.decode().split("\n")
            if code_lines:
                method_declaration = "\n".join(code_lines[:2]) 
                method_declarations.append(method_declaration)
        else:
            for child in node.children:
                method_declarations.extend(self._query_method_declarations(child))
        return method_declarations

    def _query_constructor_declarations(self, node: tree_sitter.Node):
        # actually taking entire constructor with code here, not just name
        constructor_declarations = []
        if node.type == self.constructor_declaration_identifier:
            constructor_declarations.append(node.text.decode())
        else:
            for child in node.children:
                constructor_declarations.extend(self._query_constructor_declarations(child))
        return constructor_declarations

    def _query_all_methods(self, node: tree_sitter.Node):
        methods = []
        if node.type == self.method_declaration_identifier:
            doc_comment_node = None
            if (
                node.prev_named_sibling
                and node.prev_named_sibling.type == self.doc_comment_identifier
            ):
                doc_comment_node = node.prev_named_sibling.text.decode()
            methods.append({"method": node, "doc_comment": doc_comment_node})
        else:
            for child in node.children:
                methods.extend(self._query_all_methods(child))
        return methods

    def _query_method_name(self, node: tree_sitter.Node):
        if node.type == self.method_declaration_identifier:
            code_lines = node.text.decode().split("\n")
            if code_lines:
                method_declaration = "\n".join(code_lines[:2]) 
                return method_declaration
        return None