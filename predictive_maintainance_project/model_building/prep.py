import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# ------------------------------
# Configurations
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_PATH = "hf://datasets/sudha1726/predictive-maintainanace/engine_data.csv" # Corrected dataset path
REPO_ID = "sudha1726/predictive-maintainanace"
OUTPUT_DIR = "processed_data"

# Initialize Hugging Face API
api = HfApi(token=HF_TOKEN)

# 1.Load the dataset
def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset from Hugging Face hub."""
    df = pd.read_csv(path)
    print("Dataset loaded successfully.")
    return df

# 2.Data cleaning

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform data cleaning, including outlier removal."""
    numerical_cols = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp']
    df_cleaned = df.copy()

    for col in numerical_cols:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter out the outliers
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]

    print(f"Original DataFrame shape: {df.shape}")
    print(f"Cleaned DataFrame shape: {df_cleaned.shape}")
    return df_cleaned

# 3.Split the cleaned dataset

def split_and_save(df: pd.DataFrame, target_col: str):
    """Split dataset and save train/test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    Xtrain.to_csv(f"{OUTPUT_DIR}/Xtrain.csv", index=False)
    Xtest.to_csv(f"{OUTPUT_DIR}/Xtest.csv", index=False)
    ytrain.to_csv(f"{OUTPUT_DIR}/ytrain.csv", index=False)
    ytest.to_csv(f"{OUTPUT_DIR}/ytest.csv", index=False)

    print("Train/test datasets saved locally.")
    return [
        f"{OUTPUT_DIR}/Xtrain.csv",
        f"{OUTPUT_DIR}/Xtest.csv",
        f"{OUTPUT_DIR}/ytrain.csv",
        f"{OUTPUT_DIR}/ytest.csv"
    ]

# 4.Upload the resulting train and test sets back to the hugging space dataspace

def upload_files(files: list, repo_id: str, repo_type: str = "dataset"):
    """Upload files to Hugging Face dataset repo."""
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")
    except HfHubHTTPError as e:
        print(f"Error accessing repo '{repo_id}': {e}")
        return


    for file_path in files:
        if os.path.exists(file_path):
            try:
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=os.path.basename(file_path),
                    repo_id=repo_id,
                    repo_type=repo_type
                )
                print(f" {file_path} uploaded successfully.")
            except Exception as e:
                print(f"Error uploading {file_path}: {e}")

        else:
            print(f"File not found: {file_path}")

def main():
    df = load_dataset(DATASET_PATH)
    df_cleaned = preprocess_data(df)
    files = split_and_save(df_cleaned, target_col="Engine Condition") # Changed target_col to "Engine Condition"
    upload_files(files, REPO_ID)
    print(" Preprocessing and upload complete.")

if __name__ == "__main__":
    main()
