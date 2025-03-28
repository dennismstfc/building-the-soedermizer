import pandas as pd
import numpy as np

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import EarlyStoppingCallback, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, BitsAndBytesConfig
from datasets import Dataset
from peft import LoraConfig, get_peft_model
import evaluate

from pathlib import Path
from folder_structure import ExperimentPaths

import logging
logging.basicConfig(level=logging.DEBUG)


class ModelTraining:
    """
    Class for training a model for either goal 1 or 2.
    """
    def __init__(
            self,
            training_args: Seq2SeqTrainingArguments,
            lora_config: LoraConfig,
            experiment_paths: ExperimentPaths,
            dataset_columns: list = ["non_gendered", "gendered"], # First: target sentence, second: input sentence
            model_name: str = "google/flan-t5-small",
            prefix: str = "Bringe den Satz in eine inklusive Form: "
    ):
        """
        :param training_args: The training arguments for the model training.
        :param lora_config: The LoRA configuration for the model.
        :param experiment_paths: The paths to the experiment folders.
        :param dataset_columns: The columns to be used for the dataset. First: target sentence, second: input sentence.
        :param model_name: Model checkpoint to be used for the training.
        :param prefix: The prefix to be used for the input sentences.
        """
        self.training_args = training_args
        self.lora_config = lora_config
        self.model_name = model_name

        self.model_save_path = experiment_paths.get_model_save_path()
        self.results_path = experiment_paths.get_results_path()

        train_data_path = experiment_paths.get_train_data_path()
        eval_data_path = experiment_paths.get_eval_data_path()
        
        # Make sure the train and eval paths exist
        if not train_data_path.exists():
            raise FileNotFoundError(f"Cannot find train data at {train_data_path}.")
        
        if not eval_data_path.exists():
            raise FileNotFoundError(f"Cannot find evaluation data at {eval_data_path}.")

        self.train_data_path = train_data_path
        self.eval_data_path = eval_data_path 

        self.losses = []
        self.dataset_columns = dataset_columns
        self.prefix = prefix

    @staticmethod
    def load_dataset(path: Path, dataset_columns: list) -> Dataset:
        """
        Load the dataset from the given path and return it as a Hugging Face Dataset. Selects
        only the columns that are necessary for the training.
        :param path: The path to the dataset.
        :param dataset_columns: The columns to be used for the dataset.
        :return: The dataset as a Hugging Face Dataset.
        """

        df = pd.read_csv(path)
        df = df[dataset_columns]
        df = df.dropna()  # Drop rows with missing values
        return Dataset.from_pandas(df)

    def train_model(self) -> None:
        """
        Start the training of the model.
        """
        train_dataset = self.load_dataset(self.train_data_path, self.dataset_columns)
        eval_dataset = self.load_dataset(self.eval_data_path, self.dataset_columns)

        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, legacy_format=False)

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        model = T5ForConditionalGeneration.from_pretrained(
            self.model_name, 
            quantization_config=quantization_config,
            low_cpu_mem_usage=True)

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, 
            model=model, 
            label_pad_token_id=-100, 
            pad_to_multiple_of=8
            )

        # Load the PEFT veriosn of the model
        model = get_peft_model(model, self.lora_config)
        model.print_trainable_parameters()

        train_dataset = train_dataset.map(self.__preprocess_data, batched=True)
        eval_dataset = eval_dataset.map(self.__preprocess_data, batched=True)

        early_stopping = EarlyStoppingCallback(early_stopping_patience=5)

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[early_stopping],
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )

        trainer.train()
 
        model.save_pretrained(self.model_save_path)
        self.tokenizer.save_pretrained(self.model_save_path)

    def __preprocess_data(self, examples) -> dict:
        """
        Preprocessing method for the dataset in which tokenization and further
        preprocessing steps are performed.
        :param examples: A dictionary containing the input and target sentences.
        :return: A dictionary containing the tokenized input and target sentences.
        """
        input_column = self.dataset_columns[1]
        target_column = self.dataset_columns[0]

        inputs = [self.prefix + sentence for sentence in examples[input_column]] # Use the gendered sentences as input
        targets = examples[target_column] # and try to predict the non-gendered sentences

        model_inputs = self.tokenizer(
            inputs, max_length=512, truncation=True)
        
        labels = self.tokenizer(
            targets, max_length=512, truncation=True)
        
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["labels"] = [
            label if label != self.tokenizer.pad_token_id else -100 for label in model_inputs["labels"]
        ]

        return model_inputs
    
    
if __name__ == "__main__":
    lora_config = LoraConfig(
        r=8,  # Low-rank dimension
        lora_alpha=16,  # Scaling factor
        bias="none",  # Bias strategy
        task_type="SEQ_2_SEQ_LM" # OMG THIS WAS THE MISTAKE
    )

    # Goal 1: gender-sensitive -> generic masculine/feminine
    experiment_paths = ExperimentPaths(
        experiment_name="flan_t5_finetuning_correlaid", 
        data_folder=Path("data", "standard")
        )
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=experiment_paths.get_output_path(),
        eval_strategy="steps",
        eval_steps=1000,
        logging_strategy="steps",
        logging_steps=100,
        save_steps= 1000,
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2000,
        weight_decay=0.01,
        save_total_limit=1,
        logging_dir=experiment_paths.get_logs_path(),
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    model_training_g1 = ModelTraining(
        training_args,
        lora_config,
        experiment_paths,
        prefix="Bringe den Satz in eine generische Maskuline/Feminine Form: "
        )
    
    model_training_g1.train_model()


    # Goal 2: gender-sensitive -> gender-inclusive
    experiment_paths = ExperimentPaths(
        experiment_name="flan_t5_finetuning_inclusive_form", 
        data_folder=Path("data", "inclusive_form")
        )

    training_args = Seq2SeqTrainingArguments(
        output_dir=experiment_paths.get_output_path(),
        eval_strategy="steps",
        eval_steps=1000,
        logging_strategy="steps",
        logging_steps=100,
        save_steps= 1000,
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2000,
        weight_decay=0.01,
        save_total_limit=1,
        logging_dir=experiment_paths.get_logs_path(),
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    model_training_g2 = ModelTraining(
        training_args, 
        lora_config, 
        experiment_paths, 
        dataset_columns=["inclusive_form", "gendered"],
        prefix = "Bringe den Satz in eine inklusive Form: "
        )

    model_training_g2.train_model()