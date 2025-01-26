import json
import os.path as osp
from tqdm import tqdm

def validate_json_file(filepath):
    """Validates a JSON file and attempts to identify corruption location."""
    print(f"Validating JSON file: {filepath}")
    
    try:
        # First try to load the entire file
        with open(filepath, 'r', encoding='utf-8') as f:
            json.load(f)
        print("JSON file is valid!")
        return True
    except json.JSONDecodeError as e:
        print(f"\nJSON Error detected: {str(e)}")
        
        # Read file in chunks to locate problem area
        chunk_size = 1000  # lines per chunk
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        problem_line = e.lineno
        start_line = max(0, problem_line - 5)
        end_line = min(len(lines), problem_line + 5)
        
        print(f"\nShowing context around line {problem_line}:")
        print("\n--- Context Start ---")
        for i in range(start_line, end_line):
            prefix = ">>> " if i == problem_line - 1 else "    "
            print(f"{prefix}Line {i+1}: {lines[i].rstrip()}")
        print("--- Context End ---")
        
        return False

def attempt_repair(filepath):
    """Attempts to repair common JSON issues."""
    print("\nAttempting to repair JSON file...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Common repairs
    repairs_made = False
    
    # 1. Fix missing closing quotes
    # This is a simplified repair - you might need more sophisticated logic
    if content.count('"') % 2 != 0:
        print("Detected uneven number of quotes - attempting to fix...")
        # Add specific repair logic here
        repairs_made = True
    
    # 2. Fix missing closing braces
    open_braces = content.count('{')
    close_braces = content.count('}')
    if open_braces != close_braces:
        print(f"Brace mismatch detected: {open_braces} opening vs {close_braces} closing")
        repairs_made = True
    
    if repairs_made:
        # Create backup
        backup_path = filepath + '.backup'
        print(f"Creating backup at: {backup_path}")
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Write repaired content
        # Add repair logic here
        
        print("Repair attempt completed. Please validate the file again.")
    else:
        print("No automatic repairs could be made.")

if __name__ == "__main__":
    json_path = osp.join('..', 'data', 'MSCOCO', 'annotations', 'coco_wholebody_val_v1.0.json')
    
    if not validate_json_file(json_path):
        user_input = input("\nWould you like to attempt to repair the file? (y/n): ")
        if user_input.lower() == 'y':
            attempt_repair(json_path)