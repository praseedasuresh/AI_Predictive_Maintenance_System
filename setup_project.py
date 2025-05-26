import os
import sys
import shutil

def create_directory(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")

def main():
    """Create project directory structure."""
    # Define base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define directories to create
    directories = [
        os.path.join(base_dir, "data"),
        os.path.join(base_dir, "data", "raw"),
        os.path.join(base_dir, "data", "processed"),
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, "notebooks"),
        os.path.join(base_dir, "src"),
        os.path.join(base_dir, "src", "data_processing"),
        os.path.join(base_dir, "src", "model"),
        os.path.join(base_dir, "src", "evaluation"),
        os.path.join(base_dir, "src", "utils"),
        os.path.join(base_dir, "results"),
        os.path.join(base_dir, "tests"),
    ]
    
    # Create each directory
    for directory in directories:
        create_directory(directory)
    
    # Create __init__.py files in each src directory
    src_dirs = [d for d in directories if d.startswith(os.path.join(base_dir, "src"))]
    for src_dir in src_dirs:
        init_file = os.path.join(src_dir, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write("# Initialize package\n")
            print(f"Created file: {init_file}")
    
    print("\nProject structure setup complete!")
    print("Run 'pip install -r requirements.txt' to install dependencies.")

if __name__ == "__main__":
    main()
