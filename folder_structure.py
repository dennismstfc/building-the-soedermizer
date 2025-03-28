from datetime import datetime
from pathlib import Path
import os

class ExperimentPaths:
    """
    Class for managing the directory structure of an experiment.
    """
    def __init__(
            self, 
            experiment_name: str, 
            data_folder: Path,
            base_dir: Path = Path("experiments")):
        """
        :param experiment_name: Name of the experiment (used in directory structure).
        :param data_folder: The folder containing the data for the experiment.
        :param base_dir: The base directory to store all experiment-related files.
        """
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        
        # Generate base directory for the experiment
        self.experiment_dir = self.base_dir / experiment_name
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Create experiment directory if it doesn't exist
        self._create_experiment_directories()
        self.data_folder = data_folder

    def _create_experiment_directories(self):
        """
        Create the necessary directories for the experiment.
        """
        self.model_dir = self.experiment_dir / self.timestamp / "model"
        self.results_dir = self.experiment_dir / self.timestamp / "results"
        self.logs_dir = self.experiment_dir / self.timestamp / "logs"
        self.output_dir = self.experiment_dir / self.timestamp / "output"

        # Create directories if they do not exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def get_model_save_path(self) -> Path:
        """
        :return: The path for saving the model.
        """
        return self.model_dir

    def get_results_path(self) -> Path:
        """
        :return: The path for saving the results.
        """
        return self.results_dir

    def get_logs_path(self) -> Path:
        """
        :return: The path for saving the logs.
        """
        return self.logs_dir

    def get_output_path(self) -> Path:
        """
        :return: The path for saving the output.
        """
        return self.output_dir

    def get_experiment_base_path(self) -> Path:
        """
        :return: The base path for the experiment.
        """
        return self.experiment_dir

    def get_timestamp(self) -> str:
        """
        :return: The timestamp of the experiment.
        """

        return self.timestamp

    def get_train_data_path(self) -> Path:
        """
        :return: The path for the training data.
        """
        return self.data_folder / "train.csv"
    
    def get_eval_data_path(self) -> Path:
        """	
        :return: The path for the evaluation data.	
        """
        return self.data_folder / "eval.csv"
    
    def get_test_data_path(self) -> Path:
        """"
        :return: The path for the test data.
        """
        return self.data_folder / "test.csv"