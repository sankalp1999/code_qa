from tree_sitter_languages import get_parser

# Initialize parser and read code
parser = get_parser("python")
code = '''
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def calculate_area(self):
        """Calculate the area of the rectangle."""
        return self.width * self.height
        
my_rectangle = Rectangle(5, 3)
area = my_rectangle.calculate_area()
'''

# Parse into AST
tree = parser.parse(bytes(code, "utf8"))


def extract_classes_and_methods(node):
    results = {
        'classes': [],
        'methods': []
    }
    
    def traverse_tree(node):
        # Extract class definitions
        if node.type == "class_definition":
            class_name = node.child_by_field_name("name").text.decode('utf8') 
            class_code = node.text.decode('utf8')
            results['classes'].append({
                'name': class_name,
                'code': class_code
            })
            
        # Extract method definitions
        elif node.type == "function_definition":
            method_name = node.child_by_field_name("name").text.decode('utf8')
            method_code = node.text.decode('utf8')
            results['methods'].append({
                'name': method_name,
                'code': method_code
            })
            
        # Recursively traverse children
        for child in node.children:
            traverse_tree(child)
    
    traverse_tree(node)
    return results

# Use the extraction function
extracted = extract_classes_and_methods(tree.root_node)

# Print results
for class_info in extracted['classes']:
    print(f"\nFound class {class_info['name']}:")
    print(class_info['code'])

for method_info in extracted['methods']:
    print(f"\nFound method {method_info['name']}:")
    print(method_info['code'])