from tree_sitter_languages import get_parser, get_language

# Initialize the parser and language for Python
parser = get_parser("python")
language = get_language("python")

# Sample Python code to parse
code = '''
class Rectangle:
    """Class representing a rectangle."""
    def __init__(self, width, height):
        """Initialize the rectangle with width and height."""
        self.width = width
        self.height = height
    
    def calculate_area(self):
        """Calculate the area of the rectangle."""
        return self.width * self.height
        
my_rectangle = Rectangle(5, 3)
area = my_rectangle.calculate_area()
'''

# Define Tree-sitter queries using capture tags
# Query for class definitions, capturing the name and definition
class_query = language.query("""
    (class_definition
        name: (identifier) @class.name
    ) @class.def
""")

# Query for function (method) definitions, capturing the name and definition
method_query = language.query("""
    (function_definition
        name: (identifier) @method.name
    ) @method.def
""")

def extract_classes_and_methods(root_node):
    results = {
        'classes': [],
        'methods': []
    }
    
    # Extract classes
    for match in class_query.matches(root_node):
        captures = {name: node for node, name in match.captures}
        class_name = captures['class.name'].text.decode('utf8')
        class_code = captures['class.def'].text.decode('utf8')
        results['classes'].append({
            'name': class_name,
            'code': class_code
        })
    
    # Extract methods
    for match in method_query.matches(root_node):
        captures = {name: node for node, name in match.captures}
        method_name = captures['method.name'].text.decode('utf8')
        method_code = captures['method.def'].text.decode('utf8')
        results['methods'].append({
            'name': method_name,
            'code': method_code
        })
    
    return results

# Parse the code into a syntax tree
tree = parser.parse(bytes(code, "utf8"))
# Extract classes and methods from the syntax tree
extracted = extract_classes_and_methods(tree.root_node)

# Print the results
for class_info in extracted['classes']:
    print(f"\nFound class {class_info['name']}:")
    print(class_info['code'])
    
for method_info in extracted['methods']:
    print(f"\nFound method {method_info['name']}:")
    print(method_info['code'])
