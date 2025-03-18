import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def save_into_excel(
        dataframe: pd.DataFrame, path: Path) -> None:
    """
    Function that takes a dataframe and saves it into an excel file for manual analysis.
    :param dataframe: pandas dataframe with true and predicted values
    :param path: path to save the excel file
    """    
    transformed_df = dataframe.copy()

    transformed_df = transformed_df[["true_value", "predicted", "difference", "differing_words"]]
    transformed_df["true_value"] = transformed_df["true_value"].apply(lambda x: x.sentence) # inserting the actual sentence, not the object
    transformed_df["predicted"] = transformed_df["predicted"].apply(lambda x: x.sentence) 

    # Put an empty column in the middle called "is_equal"
    transformed_df.insert(2, "is_equal", "")
    transformed_df.to_excel(path)


class Sentence:
    def __init__(self, sentence: str) -> None:
        sentence_raw = sentence.strip()
        self.sentence = sentence_raw.replace("\n", " ")
        
        lower = self.sentence.lower()
        self.lower = lower.split()


    def __sub__(self, other) -> int:
        """
        Checks the difference between two sentences. Diff is calculated as the symmetric
        difference between the two sentences. The substraction operator returns the length 
        of the difference. If the difference is 2, then most likely another gender is used 
        in the prediction sentence. E.g. "Der Bäcker hat gebacken" (true value) vs. "Die Bäckerin" 
        (predicted value) would return 2. Since the translation task was designed to not 
        differentiate between the maskulin and feminin form, a difference of 2 is acceptable.
        :param other: Another Sentence object
        """
        sentence_a = self.lower
        sentence_b = other.lower

        diff = set(sentence_a) ^ set(sentence_b)
        return len(diff)
    
    def get_difference(self, other) -> set:
        """
        Checks the difference between two sentences. 
        :param other: Another Sentence object
        """
        sentence_a = self.lower
        sentence_b = other.lower

        return set(sentence_a) ^ set(sentence_b)

    def __eq__(self, other):
        """
        Checks if two sentences are equal.
        :param other: Another Sentence object
        """
        return self.lower == other.lower


if __name__ == '__main__':
    sentence_1 = Sentence("This is a test sentence")
    sentence_2 = Sentence("This is a test example ")
    print(sentence_1 - sentence_2)
    print(sentence_1 == sentence_2)

    # Load the dataset
    dataset_1_subfolder = Path("experiments", "flan_t5_finetuning_correlaid", "2025-01-09_10-37-49", "results")
    dataset_1_path = Path(dataset_1_subfolder, "results.csv")
    dataset_1 = pd.read_csv(dataset_1_path)
    dataset_1.rename(columns={'non_gendered': 'true_value'}, inplace=True)

    dataset_1["predicted"] = dataset_1["predicted"].apply(lambda x: Sentence(x))
    dataset_1["true_value"] = dataset_1["true_value"].apply(lambda x: Sentence(x))
    
    # Analysis
    dataset_1["difference"] = dataset_1.apply(lambda x: x["true_value"] - x["predicted"], axis=1)
    dataset_1["equal"] = dataset_1.apply(lambda x: x["true_value"] == x["predicted"], axis=1)
    dataset_1["differing_words"] = dataset_1.apply(lambda x: x["true_value"].get_difference(x["predicted"]), axis=1)

    # Visualize the results
    dataset_1["difference"].value_counts().sort_index().plot(kind='bar')
    plt.xlabel("Difference of words")
    plt.ylabel("Amount of sentences")
    plt.title("Difference of words for translation task no. 1")
    plt.tight_layout()

    path = Path(dataset_1_subfolder, "dataset_1_hist_word_differences.png")
    plt.savefig(path)

    # Take all that are not equal
    dataset_1_not_equal = dataset_1[dataset_1["equal"] == False]
    dataset_1_not_equal.sort_values(by="difference", ascending=False, inplace=True)

    # Save the results into an excel file
    path = Path(dataset_1_subfolder, "dataset_1_results.xlsx")
    save_into_excel(dataset_1_not_equal, path)