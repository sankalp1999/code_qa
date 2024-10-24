#!/bin/bash

# Set the default folder path
folder_path="/Users/sankalp/Desktop/my-sweagent/SWE-agent" 
echo "Processing the entire directory at $folder_path..."

# Run scripts with the folder_path
python preprocessing.py "$folder_path"
python create_tables.py "$folder_path"

echo "Processing complete."
