import argparse
import sys
import os

# Add the project root directory to python path so it runs from anywhere
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import run_training_pipeline

def main():
    parser = argparse.ArgumentParser(
        description="Run Lending Club Loan Default Prediction pipeline (Preprocessing -> Training -> Evaluation)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the pipeline YAML configuration file (default: config/config.yaml)"
    )
    
    args = parser.parse_args()
    
    try:
        run_training_pipeline(config_path=args.config)
    except FileNotFoundError as e:
        print(f"\n[ERROR] Missing Files: {e}")
        print("Please resolve the file paths and try again.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
