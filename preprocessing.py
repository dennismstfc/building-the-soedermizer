import pandas as pd
from pathlib import Path
from enum import Enum
from tqdm import tqdm
from standard_values import *

from data_enhancer import find_inclusive_form
from sentence_structure import Sentence

import random
random.seed(42) # For reproducibility


class Mode(Enum):
    STANDARD = "standard"
    INCLUSIVE_FORM = "inclusive_form"


class DatasetCreator:
    def __init__(
            self, 
            raw_data_path: Path,
            save_folder: Path,
            split_data: bool = True,
            split_ratio: tuple = (0.8, 0.1, 0.1),
            shuffle: bool = True
            ):
        """
        :param raw_data_path: The path to the raw data to process.
        :param save_folder: The folder to save the processed data.
        :param split_data: Whether to split the data into train, eval, and test sets.
        :param split_ratio: The ratio to split the data into train, eval, and test sets.
        :param shuffle: Whether to shuffle the data before splitting.
        """
        
        if not save_folder.exists():
            save_folder.mkdir(parents=True)

        self.save_folder = save_folder

        if not raw_data_path.exists():
            raise FileNotFoundError(f"Cannot find data to process at {raw_data_path}.")

        self.raw_data = pd.read_csv(raw_data_path)

        if split_data:
            if sum(split_ratio) != 1:
                raise ValueError("The split ratio must sum up to 1.")
                    
            self.ratio = split_ratio
            self.shuffle = shuffle
            self.split_data = split_data
    
    def _split_data(self, data: pd.DataFrame, mode_folder: Path):
        """
        :param data: The dataset to split as a pandas DataFrame.
        :param mode_folder: The folder where split datasets will be saved.
        """
        if self.shuffle:
            data = data.sample(frac=1).reset_index(drop=True)

        train_size = int(len(data) * self.ratio[0])
        eval_size = int(len(data) * self.ratio[1])

        train_df = data[:train_size]
        eval_df = data[train_size:train_size + eval_size]
        test_df = data[train_size + eval_size:]

        # Save splits in mode-specific folder
        train_data_path = mode_folder / "train.csv"
        eval_data_path = mode_folder / "eval.csv"
        test_data_path = mode_folder / "test.csv"

        train_df.to_csv(train_data_path, index=False)
        eval_df.to_csv(eval_data_path, index=False)
        test_df.to_csv(test_data_path, index=False)

        print(f"Train: {len(train_df)}")
        print(f"Eval: {len(eval_df)}")
        print(f"Test: {len(test_df)}")

    def _generate_standard_dataset(self):
        """
        Generate the standard dataset with all possible combinations of gendered and non-gendered sentences.
        (maskulinum, femininum) x (neutral, doppelpunkt, unterstrich, sternchen, schraegstrich, grossbuchstaben) 
        """
        all_combinations = pd.DataFrame()
        
        for _, row in self.raw_data.iterrows():
            sentence = Sentence(row)
            all_combinations = pd.concat([
                all_combinations, sentence.subset_all_combinations()], ignore_index=True)

        all_combinations = all_combinations.dropna()

        # Save full dataset in mode-specific folder
        mode_folder = self.save_folder / "standard"
        mode_folder.mkdir(parents=True, exist_ok=True)

        full_dataset_path = mode_folder / "full_dataset.csv"
        all_combinations.to_csv(full_dataset_path, index=False)

        # Split the data and save it in the same mode-specific folder
        if self.split_data:
            self._split_data(all_combinations, mode_folder)
    
    def _generate_long_form_dataset(self):
        """
        Generate the long form dataset with the gendered sentence and the inclusive sentence.
        """
        final_data = pd.DataFrame(columns=["index", "gendered", "inclusive_form"])

        # Save full dataset in mode-specific folder
        mode_folder = self.save_folder / "inclusive_form"
        mode_folder.mkdir(parents=True, exist_ok=True)

        full_dataset_path = mode_folder / "full_dataset.csv"
        final_data.to_csv(full_dataset_path, index=False)  # Save the empty dataframe for continuous appending

        for idx, row in tqdm(self.raw_data.iterrows()):
            final_data = pd.read_csv(full_dataset_path)
            
            # Check if the sentence is already in the dataset, if so, skip to save GPT requests
            if int(idx) in final_data["index"].values:
                continue

            sentence = Sentence(row)
            random_choice = random.choice(sentence.gendered_sentences)

            tmp_df = pd.DataFrame({
                "index": [idx],
                "gendered": [random_choice],
                "inclusive_form": [find_inclusive_form(random_choice)]
            })

            final_data = pd.concat([final_data, tmp_df], ignore_index=True)
            final_data.to_csv(full_dataset_path, index=False)

        # Split the data and save it in the same mode-specific folder
        if self.split_data:
            self._split_data(final_data, mode_folder)

    def generate_dataset(self, mode: Mode = Mode.STANDARD) -> None:    
        """
        Generate the dataset based on the mode.
        :param mode: The mode of the dataset to generate. Select standard for all possible combinations of gendered 
        and non-gendered sentences. Select inclusive for the gendered sentence with longer/inclusive formulations.
        """
        if mode not in Mode:
            raise ValueError(f"Invalid mode. Expected values: {[el.value for el in Mode]}.")

        mode_folder = self.save_folder / mode.value
        mode_folder.mkdir(parents=True, exist_ok=True)

        if mode == Mode.STANDARD:
            self._generate_standard_dataset()
        else:
            self._generate_long_form_dataset()


if __name__ == "__main__":
    # Generate the standard dataset
    raw_data_path = Path("sentences.csv")
    save_folder = Path("test")

    dataset_creator = DatasetCreator(raw_data_path, save_folder)
    dataset_creator.generate_dataset(mode=Mode.STANDARD)
    dataset_creator.generate_dataset(mode=Mode.INCLUSIVE_FORM)