import pandas as pd

from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset

from standard_values import TEST_DATA_PATH
from pathlib import Path
from tqdm import tqdm


class EvaluateModel:
    def __init__(
            self, 
            model_path: Path,
            save_path: Path = Path("results"),
            batch_size: int = 16):

        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.save_path = save_path / "results.csv"
        self.batch_size = batch_size

    def eval_with_dataset(self, test_data_path: Path = TEST_DATA_PATH) -> None:
        """
        Evaluate the model with the test dataset. The method generates predictions for the non-gendered
        sentences and saves the results to a csv file.
        :param test_data_path: The path to the test dataset.
        """

        if not test_data_path.exists():
            raise FileNotFoundError(f"Cannot find evaluation data at {test_data_path}.")

        test_data_df = pd.read_csv(test_data_path)
        test_data_df = test_data_df.head(100)
        test_data_df = test_data_df[["non-gendered", "corrected"]]
        test_dataset = Dataset.from_pandas(test_data_df)
        test_dataset = test_dataset.map(self.__preprocess_data, batched=True)

        predictions = []
        for i in tqdm(range(0, len(test_dataset), self.batch_size)):
            batch = test_dataset[i:i+self.batch_size]
            batch_predictions = self.__generate_predictions(batch)
            predictions.extend(batch_predictions)

        results = pd.DataFrame({
            "non-gendered": test_data_df["non-gendered"],
            "corrected": test_data_df["corrected"],
            "predicted": predictions
        })

        results.to_csv(self.save_path, index=False)

    def __preprocess_data(self, examples) -> dict:
        """
        Preprocess the data for the model.
        """
        inputs = examples["non-gendered"]
        model_inputs = self.tokenizer(
            inputs, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
        return model_inputs

    def __generate_predictions(self, examples) -> dict:
        """
        Generate predictions for the model.
        """
        inputs = self.tokenizer(examples["non-gendered"], return_tensors="pt", max_length=512, truncation=True, padding="max_length")
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
    experiment_path = Path("experiments", "flan_t5_finetuning", "2025-01-08_11-26-55")
    model_path = Path(experiment_path, "model")
    save_path = Path(experiment_path, "results")

    eval_model = EvaluateModel(model_path, save_path)

    test_sentence = "Bringe den Satz in eine ungenderte Form: Die Lehrperson ist cool."
    print("Predicted sentence:", eval_model.eval_with_sentence(test_sentence))

#    eval_model.eval_with_dataset()
