import os

import pandas as pd
import numpy as np

SOURCE = "data/"
OUT_PATH = "data/final/"

if __name__ == "__main__":
    files = os.listdir(SOURCE)
    for file in files:
        if file.endswith(".tsv"):
            df = pd.read_csv(SOURCE + file, sep="\t")
            df = df[df["label"] == 1]
            df.to_csv(OUT_PATH + file, sep="\t")
            
