#!/bin/bash

# User inputs
read -p "Enter the language (options: python, java, rust, javascript): " language
read -p "Enter the folder path: " folder_path

# Confirm the inputs
echo "Processing the entire directory at $folder_path with language set to $language..."

# Run scripts with the folder_path
python preprocessing.py "$language" "$folder_path"
python llm_comments.py "$language" "$folder_path"
python create_tables.py "$language" "$folder_path"

echo "Processing complete."
