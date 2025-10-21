import os

IGNORE_DIRS = {'.git','data', '__pycache__', 'venv', '.idea', '.mypy_cache', '.pytest_cache'}

def print_tree(start_path='.', prefix=''):
    entries = sorted(os.listdir(start_path))
    entries = [e for e in entries if e not in IGNORE_DIRS]

    for i, entry in enumerate(entries):
        path = os.path.join(start_path, entry)
        connector = "└── " if i == len(entries) - 1 else "├── "

        if os.path.isdir(path):
            print(f"{prefix}{connector}{entry}/")
            extension = "    " if i == len(entries) - 1 else "│   "
            print_tree(path, prefix + extension)
        else:
            print(f"{prefix}{connector}{entry}")

if __name__ == "__main__":
    print_tree('.')
