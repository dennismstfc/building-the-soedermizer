import pandas as pd
from pathlib import Path

from standard_values import *


class Sentence:
    """
    A class to represent a sentence of the dataset. It contains
    a method to generate all possible combinations of gendered and non-gendered sentences.
    """
    def __init__(self, row: dict):
        # Non-gendered sentences
        maskulinum = row["maskulinum"]
        femininum = row["femininum"]
        self.non_gendered = [maskulinum, femininum]

        # Gendered
        neutral = row["neutral"]
        doppelpunkt = row["inkl_Doppelpunkt"]
        unterstrich = row["inkl_Unterstrich"]
        sternchen = row["inkl_Sternchen"]
        schraegstrich = row["inkl_Schrägstrich"]
        grossbuchstaben = row["inkl_Grossbuchstaben"]
        self.gendered = [neutral, doppelpunkt, unterstrich, sternchen, schraegstrich, grossbuchstaben]

    def subset_all_combinations(self) -> pd.DataFrame:
        """
        Return a dataframe with all possible combinations of gendered and non-gendered sentences.
        """
        df = pd.DataFrame()

        gendered = []
        non_gendered = []

        for gendered_sentence in self.gendered:
            for non_gendered_sentence in self.non_gendered:
                gendered.append(gendered_sentence)
                non_gendered.append(non_gendered_sentence)
        
        df["gendered"] = gendered
        df["non_gendered"] = non_gendered

        return df
    

def generate_dataset(
        data_path: Path, 
        save_folder: Path = DATA_PATH, 
        split_ratio: tuple = (0.8, 0.1, 0.1)
        ) -> None:

    # Check if the split ratio is valid
    if sum(split_ratio) != 1:
        raise ValueError("The split ratio must sum up to 1.")

    # Make sure that data_path exists
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find data at {data_path}.")

    save_path = Path(save_folder, "all_combinations.csv")

    data = pd.read_csv(data_path)

    all_combinations = pd.DataFrame()
    for _, row in data.iterrows():
        sentence = Sentence(row)
        all_combinations = pd.concat([
            all_combinations, sentence.subset_all_combinations()], ignore_index=True)
        
    all_combinations = all_combinations.dropna()
    all_combinations.to_csv(save_path, index=False)

    # shuffle the dataset
    all_combinations = all_combinations.sample(frac=1).reset_index(drop=True)

    # Split the dataset into train, test and validation
    train_size = int(len(all_combinations) * split_ratio[0])
    eval_size = int(len(all_combinations) * split_ratio[1])

    train_df = all_combinations[:train_size]
    eval_df = all_combinations[train_size:train_size + eval_size]
    test_df = all_combinations[train_size + eval_size:]

    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    eval_df.to_csv(EVAL_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    print("Train: ", len(train_df))
    print("Eval: ", len(eval_df))
    print("Test: ", len(test_df))

    
if __name__ == "__main__":
    data_path = Path("data", "sentences.csv")
    generate_dataset(data_path)