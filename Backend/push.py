import os
import shutil
import subprocess
from dotenv import load_dotenv
import errno
import time

# === CONFIGURATION ===
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set. Please set it or use huggingface-cli login.")
huggingface_repo_url = f"https://{hf_token}@huggingface.co/spaces/Tanmay0483/ArogyaAI"
local_clone_folder = "hf_space_clone"
your_project_folder = "."  # Root of your actual code
binary_patterns = ["*.pkl", "*.h5", "*.keras", "*.bin", "*.pth", "*.png"]  # Track binary files with LFS
essential_files = ["Dockerfile", "main.py", "requirements.txt"]  # Essential files
essential_dirs = ["router", "cluster_model", "notebooks/models", "notebooks/vaccine_models", "notebooks/vaccine_scalers"]  # Model directories

# === STEP 1: Clone the Space repo ===
def remove_directory(path, retries=5, delay=3):
    for attempt in range(retries):
        try:
            if os.path.exists(path):
                print(f"🧹 Removing existing clone folder (attempt {attempt + 1}/{retries})...")
                shutil.rmtree(path, ignore_errors=False)
            return
        except PermissionError as e:
            print(f"⚠️ PermissionError: {e}. Retrying after {delay} seconds...")
            time.sleep(delay)
        except OSError as e:
            if e.errno != errno.ENOENT:
                print(f"⚠️ Error removing directory: {e}")
                exit(1)
    print(f"⚠️ Failed to remove {path} after {retries} attempts.")
    print("Please try the following:")
    print("1. Close any programs accessing the folder (e.g., Git, VS Code, File Explorer).")
    print("2. Manually delete the folder with: rmdir /s /q hf_space_clone")
    print("3. Run this script in PowerShell as Administrator.")
    exit(1)

remove_directory(local_clone_folder)

print(f"🚀 Cloning the repo from Hugging Face...")
try:
    subprocess.run(["git", "clone", huggingface_repo_url, local_clone_folder], check=True)
except subprocess.CalledProcessError as e:
    print(f"⚠️ Failed to clone repository: {e}")
    print("Ensure the Space exists and the HF_TOKEN has write access.")
    exit(1)

# === STEP 2: Copy essential files and model directories to the repo ===
def copy_folder(src, dst):
    exclude_dirs = {'.git', '__pycache__', '.venv', 'venv', 'hf_space_clone', '.vscode', '.idea', '.ipynb_checkpoints'}
    exclude_files = {'.gitignore', '.env', '*.ipynb'}
    # Copy essential files from the root
    for file in essential_files:
        src_file = os.path.join(src, file)
        if os.path.exists(src_file):
            dst_file = os.path.join(dst, file)
            shutil.copy2(src_file, dst_file)
            print(f"Copied {src_file} to {dst_file}")

    # Copy essential directories (e.g., router, model directories)
    for dir_name in essential_dirs:
        src_dir = os.path.join(src, dir_name)
        dst_dir = os.path.join(dst, dir_name)
        if os.path.exists(src_dir):
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*exclude_dirs, *exclude_files))
            print(f"Copied directory {src_dir} to {dst_dir}")

print("📁 Copying project files...")
try:
    copy_folder(your_project_folder, local_clone_folder)
except Exception as e:
    print(f"⚠️ Error copying files: {e}")
    exit(1)

# === STEP 3: Git LFS Track binary files ===
os.chdir(local_clone_folder)
print("🔍 Setting up Git LFS for binary files...")
try:
    subprocess.run(["git", "lfs", "install"], check=True)
    for pattern in binary_patterns:
        subprocess.run(["git", "lfs", "track", pattern], check=True)
    with open(".gitattributes", "a") as f:
        for pattern in binary_patterns:
            f.write(f"{pattern} filter=lfs diff=lfs merge=lfs -text\n")
except subprocess.CalledProcessError as e:
    print(f"⚠️ Error setting up Git LFS: {e}")
    exit(1)

# === STEP 4: Commit and push to Hugging Face ===
print("📦 Adding and committing files...")
try:
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "Initial deployment of Docker-based API with main.py, router, and model files"], check=True)
except subprocess.CalledProcessError as e:
    print(f"⚠️ No changes to commit or error: {e}")
    # Continue to push even if no changes

print("☁️ Pushing to Hugging Face...")
try:
    subprocess.run(["git", "push"], check=True)
except subprocess.CalledProcessError as e:
    print(f"⚠️ Failed to push to Hugging Face: {e}")
    exit(1)

print("✅ All done! Your Docker-based API is live on Hugging Face 🚀")