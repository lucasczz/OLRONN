#!/bin/bash

# Find all Python files in the current directory and its subdirectories
find . -name "*.py" | while read file; do
  # Execute the Python file
  python "$file"
done

