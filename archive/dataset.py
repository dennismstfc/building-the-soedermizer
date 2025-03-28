import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm
import language_tool_python

from standard_values import WIKIPEDIA_SENTENCES_PATH, DATA_PATH
from dictionary import GenderDictionary


class DatasetCreator:
    """
    Class for creating the dataset by translating the non-gendered sentences to gendered sentences with
    grammar correction using the language tool. The dataset is then split into train, test and validation.
    """

    def __init__(
            self, 
            split: tuple = (0.8, 0.1, 0.1),
            dataset_name: str = "final_data", 
            chunk_size: int = 10_000):
        """
        :param split: Tuple containing the split size for the training, evaluation and test dataset.
        :param dataset_name: The name of the dataset.
        :param chunk_size: The size of the chunks to split the dataset into to avoid memory issues.
        """
        if sum(split) != 1:
            raise ValueError("The split_size must sum up to 1.")
        
        self.split = split
        
        self.__load_data()
        self.save_path = DATA_PATH / (dataset_name + ".csv")
        self.chunk_size = chunk_size

    def __load_data(self):
        # Check if the wikipedia sentences file exists
        if not WIKIPEDIA_SENTENCES_PATH.exists():
            print("Cannot find existing wikipedia sentences file. Downloading data from kaggle...")
            self.__download_data_from_kaggle()
        else:
            print("Found existing wikipedia sentences file. Loading data...")

        # Load the wikipedia sentences
        with open(WIKIPEDIA_SENTENCES_PATH, 'r', encoding='utf-8') as file:
            self.wikipedia_sentences = file.readlines()
    
    def __download_data_from_kaggle(self):
        api = KaggleApi()
        api.authenticate()
        
        dataset = "bminixhofer/8m-german-sentences-from-wikipedia"
        api.dataset_download_files(dataset, path='data/', unzip=True)

    def split_dataset(self):
        """
        Split the dataset into train, test and validation with the given sizes.
        """
        print(f"Splitting dataset into {self.split[0] * 100}% training, {self.split[1] * 100}% evaluation and {self.split[2] * 100}% test...")
        # Load the dataset
        df = pd.read_csv(self.save_path)
        df = df[df["was_gendered"] == True]

        # Split the dataset into train, test and validation
        train_size = int(len(df) * self.split[0])
        eval_size = int(len(df) * self.split[1])

        train_df = df[:train_size]
        eval_df = df[train_size:train_size + eval_size]
        test_df = df[train_size + eval_size:]

        train_df.to_csv(DATA_PATH / "train.csv", index=False)
        eval_df.to_csv(DATA_PATH / "eval.csv", index=False)
        test_df.to_csv(DATA_PATH / "test.csv", index=False)

        print("Train: ", len(train_df))
        print("Eval: ", len(eval_df))
        print("Test: ", len(test_df))
    
    def create_dataset(self):
        """
        Create the dataset by translating the non-gendered sentences to gendered sentences with 
        grammar correction using the language tool. The dataset is then split into train, test and
        validation.
        """
        df = pd.DataFrame(self.wikipedia_sentences, columns=["non-gendered"])
        gender_dict = GenderDictionary()

        # Load the grammar tool for grammar correction, e.g. when "Der Lehrer ist cool" ->
        # "Der Lehrkraft ist cool" results from the naive translation approach, the grammar 
        # tool will (hopefully) fix this mistake and use the correct article.
        grammar_tool = language_tool_python.LanguageTool('de')

        # Check if the final_data.csv file already exists
        if self.save_path.exists():
            existing_df = pd.read_csv(self.save_path)
            last_index = existing_df.index[-1] + 1
            print(f"Resuming from index {last_index}...")
        else:
            last_index = 0

        # Split the dataframe into chunks to avoid memory issues
        print("Creating dataset...")
        for i in tqdm(range(last_index, len(df), self.chunk_size)):
            df_chunk = df[i:i + self.chunk_size].copy()

            df_chunk.loc[:, "gendered"] = df_chunk["non-gendered"].apply(lambda x: gender_dict.translate(x))
            df_chunk.loc[:, "corrected"] = df_chunk["gendered"].apply(lambda x: grammar_tool.correct(x))
            df_chunk.loc[:, "was_gendered"] = df_chunk["non-gendered"] != df_chunk["gendered"]

            df_chunk.reset_index(inplace=True)
            df_chunk.to_csv(self.save_path, index=False, mode='a', header=not self.save_path.exists())

        df.to_csv(self.save_path, index=False, mode='a', header=not self.save_path.exists())
        df = df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle the dataset

        # Split the dataset into train, test and validation
        self.split_dataset()


if __name__ == "__main__":
    dc = DatasetCreator(dataset_name="final_data_v2", chunk_size=1_000)
    dc.create_dataset()