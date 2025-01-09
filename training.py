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

TRAIN_DATA_PATH = Path("data_correl_aid", "train.csv")
EVAL_DATA_PATH = Path("data_correl_aid", "eval.csv")

class ModelTraining:
    def __init__(
            self,
            training_args: Seq2SeqTrainingArguments,
            lora_config: LoraConfig,
            experiment_paths: ExperimentPaths,
            model_name: str = "google/flan-t5-small",
            train_data_path: Path = TRAIN_DATA_PATH,
            eval_data_path: Path = EVAL_DATA_PATH
    ):
        self.training_args = training_args
        self.lora_config = lora_config
        self.model_name = model_name

        self.model_save_path = experiment_paths.get_model_save_path()
        self.results_path = experiment_paths.get_results_path()
        
        # Make sure the train and eval paths exist
        if not train_data_path.exists():
            raise FileNotFoundError(f"Cannot find train data at {train_data_path}.")
        
        if not eval_data_path.exists():
            raise FileNotFoundError(f"Cannot find evaluation data at {eval_data_path}.")

        self.train_data_path = train_data_path
        self.eval_data_path = eval_data_path

        self.losses = []

    @staticmethod
    def load_dataset(path: Path) -> Dataset:
        df = pd.read_csv(path)
        df = df[["non_gendered", "gendered"]]
        return Dataset.from_pandas(df)

    def train_model(self) -> None:
        train_dataset = self.load_dataset(self.train_data_path)
        eval_dataset = self.load_dataset(self.eval_data_path)

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
            data_collator=data_collator,
#            compute_metrics=self.__compute_metrics
        )

        trainer.train()
 
        model.save_pretrained(self.model_save_path)
        self.tokenizer.save_pretrained(self.model_save_path)

        self.log_history = trainer.state.log_history


    def __preprocess_data(self, examples) -> dict:
        prefix = "Bringe den Satz in eine ungenderte Form: "
        inputs = [prefix + sentence for sentence in examples["gendered"]] # Use the gendered sentences as input
        targets = examples["non_gendered"] # and try to predict the non-gendered sentences

        model_inputs = self.tokenizer(
            inputs, max_length=512, truncation=True)
        
        labels = self.tokenizer(
            targets, max_length=512, truncation=True)
        
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["labels"] = [
            label if label != self.tokenizer.pad_token_id else -100 for label in model_inputs["labels"]
        ]

        return model_inputs
    
    def __compute_metrics(self, eval_pred) -> dict:
        predictions, labels = eval_pred
        # Ensure labels are not padded
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        google_bleu = evaluate.load("google_bleu")
        bleu_score = google_bleu.compute(predictions=decoded_preds, references=decoded_labels)

        return {"bleu": bleu_score["score"]}
    

if __name__ == "__main__":
    lora_config = LoraConfig(
        r=8,  # Low-rank dimension
        lora_alpha=16,  # Scaling factor
        bias="none",  # Bias strategy
        task_type="SEQ_2_SEQ_LM" # OMG THIS WAS THE MISTAKE
    )

    experiment_paths = ExperimentPaths("flan_t5_finetuning_correlaid")

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

    model_training = ModelTraining(training_args, lora_config, experiment_paths)
    model_training.train_model()