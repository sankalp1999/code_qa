from treesitter import Treesitter, Language

class TreesitterJava(Treesitter):
    def __init__(self):
        super().__init__(
            Language.JAVA,
            "method_declaration",
            "identifier",
            "block_comment",
            "class_declaration",
            "constructor_declaration"
        )

class TreesitterPython(Treesitter):
    def __init__(self):
        super().__init__(
            Language.PYTHON,
            "function_definition",
            "identifier",
            "comment",
            "class_definition",
            "function_definition"
        )

class TreesitterRust(Treesitter):
    def __init__(self):
        super().__init__(
            Language.RUST,
            "function_item",
            "identifier",
            "line_comment",
            "struct_item",
            "impl_item"
        )

# all methods corresponding to a struct are in impl_item 

class TreesitterJavaScript(Treesitter):
    def __init__(self):
        super().__init__(
            Language.JAVASCRIPT,
            "function_declaration",
            "identifier",
            "comment",
            "class_declaration",
            "method_definition"
        )

# all methods including constructor are method_definition for javascript so this is not fully accurate but will serve our purposes