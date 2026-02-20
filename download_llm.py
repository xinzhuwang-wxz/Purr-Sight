import os
import argparse
import sys

def check_dependencies():
    try:
        import huggingface_hub
        return True
    except ImportError:
        print("Error: 'huggingface_hub' is not installed.")
        print("Please install it using: pip install huggingface_hub")
        return False

def download_model(repo_id, local_dir):
    from huggingface_hub import snapshot_download
    
    print(f"Starting download of {repo_id}...")
    print(f"Destination: {local_dir}")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Keep actual files for easier inspection/portability
            resume_download=True
        )
        print(f"\nSuccessfully downloaded {repo_id} to {local_dir}!")
        print(f"You can now point your 'llm_model_path' in config to: {os.path.abspath(local_dir)}")
        return True
    except Exception as e:
        print(f"\nError downloading {repo_id}: {e}")
        return False

if __name__ == "__main__":
    if not check_dependencies():
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Download LLM weights for Purr-Sight.")
    
    # Default to Qwen2.5-0.5B-Instruct as recommended
    parser.add_argument("--repo_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", 
                        help="Hugging Face repo ID (default: Qwen/Qwen2.5-0.5B-Instruct)")
    
    # Default output directory
    default_output = os.path.join("models", "Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", type=str, default=default_output, 
                        help=f"Local directory to save the model (default: {default_output})")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            print(f"Created directory: {args.output_dir}")
        except Exception as e:
            print(f"Error creating directory {args.output_dir}: {e}")
            sys.exit(1)
    
    success = download_model(args.repo_id, args.output_dir)
    if not success:
        sys.exit(1)
