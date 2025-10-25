# data_download_and_prepare.py
from datasets import load_dataset
import pandas as pd

def main():
    ds = load_dataset("takala/financial_phrasebank", "sentences_50agree")
    df = ds['train'].to_pandas()
    df.to_csv("financial_phrasebank_50agree.csv", index=False)
    print("Saved CSV with shape:", df.shape)

if __name__ == "__main__":
    main()
