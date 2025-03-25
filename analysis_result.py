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
        in the prediction sentence. E.g. "Der Bäcker hat gebacken" (true value) vs. "Die Bäckerin
        hat gebacken" (predicted value) would return 2. Since the translation task was designed 
        to not differentiate between the maskulin and feminin form, a difference of 2 is acceptable.
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
    # Testing the Sentence class
    sentence_1 = Sentence("This is a test sentence")
    sentence_2 = Sentence("This is a test example ")
    print(f"Sentence 1: {sentence_1.sentence}, Sentence 2: {sentence_2.sentence}")
    print("Symmetric word difference: ", sentence_1 - sentence_2)

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

    # Take all that are not equal
    dataset_1_not_equal = dataset_1[dataset_1["equal"] == False]
    dataset_1_not_equal.sort_values(by="difference", ascending=False, inplace=True)

    # Save the results into an excel file
    path = Path(dataset_1_subfolder, "dataset_1_results.xlsx")

    # Don't uncomment the subsequent line, since the dataset was already manually evaluated
#    save_into_excel(dataset_1_not_equal, path)

    # Load the manually evaluated dataset
    dataset_1_evaluated_path = Path(dataset_1_subfolder, "dataset_1_results.xlsx")
    dataset_1_evaluated = pd.read_excel(dataset_1_evaluated_path)

    dataset_1_evaluated = dataset_1_evaluated[["difference", "is_equal"]]
    dataset_1_evaluated["is_equal"] = dataset_1_evaluated["is_equal"].apply(lambda x: x == "X")

    max_difference = dataset_1_evaluated["difference"].max()
    min_difference = dataset_1_evaluated["difference"].min()
    difference_accuracy = {}

    for act_diff in range(min_difference, max_difference + 1):
        act_diff_df = dataset_1_evaluated[dataset_1_evaluated["difference"] == act_diff]

        # It can be the case that a certain difference does not exist within the span
        if act_diff_df.empty:
            continue
        
        # For the last four differences, take only the first 50 data points.
        # Due to the time consuming manual evaluation, only the first 50 data points were evaluated.
        # Look into the according paper for more information.
        if act_diff < 5:
            act_diff_df = act_diff_df.head(50)

        number_vals = act_diff_df.shape[0]
        act_diff_df = act_diff_df["is_equal"].value_counts(normalize=True)

        difference_accuracy[act_diff] = [number_vals, 1 - act_diff_df[False].item()]
    
    difference_accuracy[0] = [dataset_1[dataset_1["difference"] == 0].shape[0], 1.0]

    # Visulization 
    custom_x_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 19]    
    x_labels = [str(x) for x in custom_x_order]
    y_values = [difference_accuracy[x][1] for x in custom_x_order]
    sizes = [difference_accuracy[x][0] for x in custom_x_order]

    plt.figure(figsize=(8, 5))
    plt.scatter(range(len(x_labels)), y_values, s=sizes, alpha=0.5, color="royalblue")

    # Adding crosses
    for i, size in enumerate(sizes):
        if size > 100:
            plt.scatter(i, y_values[i], s=20, color="black", marker="x")

    plt.xlabel("Symmetric word difference")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per symmetric word difference")
    plt.xticks(range(len(x_labels)), x_labels)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.ylim(-0.1, 1.2)
    plt.xlim(-1.2, 12.2)

    bubble_plot_path = Path("experiments", "flan_t5_finetuning_correlaid", "2025-01-09_10-37-49", "results", "bubble_plot.pgf")
    plt.savefig(bubble_plot_path)
    plt.show()

    # Weighted average accuracy calculation
    overall_acc = 0
    for key, value in difference_accuracy.items():
        overall_acc += value[0] * value[1]
    
    overall_acc = overall_acc / sum([value[0] for value in difference_accuracy.values()])

    print(50 * "-")
    print("Dataset 1")
    print(f"Per difference accuracy: {difference_accuracy}")
    print(f"Overall accuracy: {overall_acc}")

    # Manual evaluation of the GPT generated data
    gpt_data_subfolder = Path("data", "inclusive_form", "v1")
    gpt_data_path = Path(gpt_data_subfolder, "full_dataset.csv")
    gpt_data = pd.read_csv(gpt_data_path)

    # Target value = inclusive_form
    gpt_data.rename(columns={"enhanced": "inclusive_form"}, inplace=True)
    gpt_data = gpt_data[["gendered", "inclusive_form"]]
    gpt_data.insert(1, "succeeded", "")
    gpt_data.insert(1, "changed_meaning", "")
    gpt_data.insert(1, "hallucination", "")
    
    gpt_data_save_path = Path(gpt_data_subfolder, "manual_evaluation.xlsx")
    # Don't uncomment the subsequent line, since the dataset was already manually evaluated
#    gpt_data.to_excel(save_path) 
    
    # Load the manually evaluated dataset
    gpt_data_evaluated = pd.read_excel(gpt_data_save_path)
    
    # Loading the first 50 rows of the dataset which where manually evaluated
    gpt_data_evaluated = gpt_data_evaluated.head(50)
    gpt_data_evaluated["succeeded"] = gpt_data_evaluated["succeeded"].apply(lambda x: x == "X")
    gpt_data_evaluated["changed_meaning"] = gpt_data_evaluated["changed_meaning"].apply(lambda x: x == "X")
    gpt_data_evaluated["hallucination"] = gpt_data_evaluated["hallucination"].apply(lambda x: x == "X")

    # Calculate the accuracy of the perfect translations
    succeeded_acc = gpt_data_evaluated["succeeded"].value_counts(normalize=True)[True]
    
    # Calculate the accuaracy with the hallucinations -> hallucination + succeeded
    hallucination_acc = gpt_data_evaluated["hallucination"].value_counts(normalize=True)[True] + succeeded_acc

    print("")	
    print(50 * "-")
    print("GPT-3 Data")
    print(f"Accuracy of the perfect translations: {succeeded_acc}")
    print(f"Accuracy of okay-ish translations: {hallucination_acc}")
    print(f"Percentage of the failed translations: {1 - hallucination_acc}")

    # Checking the results of the flan-t5-small which was trained with the gpt data 
    dataset_2_subfolder = Path("experiments", "flan_t5_finetuning_inclusive_form", "2025-01-26_19-33-13", "results")
    dataset_2_path = Path(dataset_2_subfolder, "results.csv")
    dataset_2 = pd.read_csv(dataset_2_path)
    
    dataset_2.rename(columns={"enhanced": "true_value"}, inplace=True)
    dataset_2.insert(2, "succeeded", "")
    dataset_2.insert(2, "failed", "")
    dataset_2.insert(2, "hallucinated", "")
    
    dataset_2_save_path = Path(dataset_2_subfolder, "manual_evaluation.xlsx")

    # Don't uncomment the subsequent line, since the dataset was already manually evaluated
#    dataset_2.to_excel(dataset_2_save_path)

    # Load the manually evaluated dataset
    dataset_2_evaluated = pd.read_excel(dataset_2_save_path)
    
    # As before, only the first 50 data points were manually evaluated
    dataset_2_evaluated = dataset_2_evaluated.head(50)
    dataset_2_evaluated["succeeded"] = dataset_2_evaluated["succeeded"].apply(lambda x: x == "X")
    dataset_2_evaluated["failed"] = dataset_2_evaluated["failed"].apply(lambda x: x == "X")
    dataset_2_evaluated["hallucinated"] = dataset_2_evaluated["hallucinated"].apply(lambda x: x == "X")

    # Calculate the accuracy of the perfect translations
    succeeded_acc = dataset_2_evaluated["succeeded"].value_counts(normalize=True)[True]

    # Calculate the accuaracy with the hallucinations -> hallucination + succeeded
    hallucination_acc = dataset_2_evaluated["hallucinated"].value_counts(normalize=True)[True] + succeeded_acc

    print("")
    print(50 * "-")
    print("Dataset 2")
    print(f"Accuracy of the perfect translations: {succeeded_acc}")
    print(f"Accuracy of okay-ish translations: {hallucination_acc}")
    print(f"Percentage of the failed translations: {1 - hallucination_acc}")