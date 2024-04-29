import os
import re
import sys
from collections import defaultdict

def extract_classes(content):
    # Regex to find class/interface/enum declarations
    pattern = re.compile(r'\b(class|interface|enum)\s+(\w+)')
    return {match.group(2) for match in pattern.finditer(content)}

def find_references(files, class_to_file):
    # Dictionary to store where each file's classes are referenced
    file_references = defaultdict(set)

    # Loop through files and search for each class
    for path, filename in files:
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
            for classname, origin_files in class_to_file.items():
                if classname in content:
                    for origin_file in origin_files:
                        if origin_file != filename:
                            file_references[origin_file].add(filename)

    return file_references

def main(directory):
    # Dictionary to map classes to their files
    class_to_file = defaultdict(set)
    all_files = []

    # Walk through the directory structure
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.java'):
                path = os.path.join(root, file)
                all_files.append((path, file))
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    classes = extract_classes(content)
                    for cls in classes:
                        class_to_file[cls].add(file)

    # Find references
    references = find_references(all_files, class_to_file)

    # Print the results
    for file, refs in references.items():
        print(f"{file} is referenced in:")
        for ref in refs:
            print(f"  - {ref}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_java_files>")
        sys.exit(1)

    main(sys.argv[1])
