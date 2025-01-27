import pandas as pd

from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset

from pathlib import Path
from tqdm import tqdm


class EvaluateModel:
    def __init__(
            self, 
            model_path: Path,
            save_path: Path = Path("results"),
            batch_size: int = 16,
            dataset_columns: list = ["non_gendered", "gendered"],
            prefix: str = "Bringe den Satz in eine ungegenderte Form: "
            ):
        """
        :param model_path: The path to the trained model.
        :param save_path: The path to save the results.
        :param batch_size: The batch size for generating predictions.
        :param dataset_columns: The columns of the dataset. The first column should be the target sentence, the second the input sentence.
        :param prefix: The prefix to add to the input sentence, being for example "Übersetze den Satz in XY: ".
        """

        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.save_path = save_path / "results.csv"
        self.batch_size = batch_size
        self.dataset_columns = dataset_columns
        self.prefix = prefix

    def eval_with_dataset(self, test_data_path: Path) -> None:
        """
        Evaluate the model with the test dataset. The method generates predictions for the non-gendered
        sentences and saves the results to a csv file.
        :param test_data_path: The path to the test dataset.
        """

        if not test_data_path.exists():
            raise FileNotFoundError(f"Cannot find evaluation data at {test_data_path}.")

        test_data_df = pd.read_csv(test_data_path)
        test_data_df = test_data_df[self.dataset_columns]
        test_data_df = test_data_df.dropna() # Drop rows with missing values

        test_dataset = Dataset.from_pandas(test_data_df)
        test_dataset = test_dataset.map(self.__preprocess_data, batched=True)

        predictions = []
        for i in tqdm(range(0, len(test_dataset), self.batch_size)):
            batch = test_dataset[i:i+self.batch_size]
            batch_predictions = self.__generate_predictions(batch)
            predictions.extend(batch_predictions)

        results = pd.DataFrame({
            self.dataset_columns[0]: test_data_df[self.dataset_columns[0]],
            self.dataset_columns[1]: test_data_df[self.dataset_columns[1]],
            "predicted": predictions
        })

        results.to_csv(self.save_path, index=False)

    def __preprocess_data(self, examples: list) -> dict:
        """
        Preprocess the data for the model.
        :param examples: The examples to preprocess.
        :return: The preprocessed examples.
        """
        inputs = [self.prefix + sentence for sentence in examples[self.dataset_columns[1]]]
        model_inputs = self.tokenizer(
            inputs, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
        return model_inputs

    def __generate_predictions(self, examples) -> dict:
        """
        Generate predictions for the model.
        """
        inputs = self.tokenizer(examples[self.dataset_columns[0]], return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # Generate predictions using the model
        outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=512)
        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return predictions

    def eval_with_sentence(self, sentence: str) -> str:
        """
        Evaluate the model with a single sentence.
        :param sentence: The sentence to evaluate.
        :return: The predicted sentence.
        """
        inputs = self.tokenizer(sentence, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # Generate predictions using the model
        outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=512)
        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return predictions[0]


if __name__ == "__main__":
    # First experiment: translation from gendered to non-gendered sentences
    experiment_path = Path("experiments", "flan_t5_finetuning_correlaid", "2025-01-09_10-37-49")

    model_path = Path(experiment_path, "model")
    save_path = Path(experiment_path, "results")

    eval_model = EvaluateModel(model_path, save_path)

    test_sentence = "Bringe den Satz in eine ungenderte Form: Die Lehrer*innen sind cool."
    print("Predicted sentence:", eval_model.eval_with_sentence(test_sentence))

    test_data_path = Path("data", "standard", "test.csv")
    eval_model.eval_with_dataset()


    # Second experiment: translation from gendered sentences into inclusive form
    experiment_path = Path("experiments", "flan_t5_finetuning_inclusive_form", "2025-01-26_19-33-13")
    model_path = Path(experiment_path, "model")
    save_path = Path(experiment_path, "results")

    eval_model = EvaluateModel(
        model_path, 
        save_path, 
        dataset_columns=["gendered", "enhanced"], 
        prefix = "Bringe den Satz in eine inklusive Form: "
        )

    test_sentence = "Bringe den Satz in eine inklusive Form: Die Lehrer*innen sind cool."
    print("Predicted sentence:", eval_model.eval_with_sentence(test_sentence))

    test_data_path = Path("data", "inclusive_form", "test.csv")
    eval_model.eval_with_dataset(test_data_path)
