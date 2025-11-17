import random
import re
from collections.abc import Mapping
from re import Pattern
from typing import Any, cast

import nltk
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from sentence_transformers.losses import DenoisingAutoEncoderLoss
from torch.utils.data import DataLoader

from tsdae.exceptions import (
    DataLoaderCreationError,
    DatasetConstraintError,
    DatasetLoadError,
    InvalidBatchSizeError,
    InvalidDatasetTypeError,
    InvalidModelInputError,
    InvalidModelNameError,
    InvalidNumWorkersError,
    InvalidRegexPatternError,
    InvalidSentencesInputError,
    InvalidTieEncoderDecoderError,
    LossCreationError,
    ModelCreationError,
    NLTKDownloadError,
    TrainingError,
    TypeMismatchError,
    ValidationError,
)
from tsdae.utils import create_unverified_https_context, get_logger

try:
    create_unverified_https_context()
    nltk.download("punkt")

except LookupError as e:
    raise NLTKDownloadError(e) from e


class TSDAE:
    """
    Learning sentence embeddings often requires a large amount of labeled data.
    However, for most tasks and domains, labeled data is seldom available and creating it is expensive.
    In this work, we present a new state-of-the-art unsupervised method based on pre-trained
    Transformers and Sequential Denoising Auto-Encoder (TSDAE) which outperforms previous approaches
    by up to 6.4 points. It can achieve up to 93.1% of the performance of in-domain supervised approaches.

    The model architecture of TSDAE is a modified encoder-decoder Transformer where the key
    and value of the cross-attention are both confined to the sentence embedding only.

    Example:
        >>> instance = TSDAE()
        >>> train_dataset = instance.load_dataset(
            Dataset="louisbrulenaudet/cgi"
        )
        >>> model = instance.train(
            train_dataset=train_dataset,
            model_name="bert-base-multilingual-uncased",
            column="output",
            output_path="output/tsdae-lemone-mbert-base"
        )
    """

    def __init__(self) -> None:
        """
        Initializes an instance of TSDAE with enhanced logging for error handling.

        Note:
            This constructor initializes the instance of TSDAE with a sophisticated logging configuration to
            facilitate comprehensive error handling. The logging level is set to ERROR, ensuring that
            only critical messages and above are recorded. The logger instance is created using the
            standardized logger utility, providing clear identification in logs.

        Example:
            >>> instance = TSDAE()
        """
        # Create a logger with the class name using the standardized utility
        self.logger = get_logger(__name__)

    def load_dataset_from_hf(
        self,
        dataset: list[str] | str,
        streaming: bool = True,
        split: str | None = "train",
    ) -> (
        list[Dataset | IterableDataset | DatasetDict | IterableDatasetDict]
        | Dataset
        | IterableDataset
        | DatasetDict
        | IterableDatasetDict
        | None
    ):
        """
        Downloads dataset(s) specified either in a list or as a string and returns the loaded datasets.

        Args:
            dataset: A list containing the names of datasets to be downloaded or a
                string representing a single dataset name.
            streaming: Determines if datasets are streamed. Default is True.
            split: The split of the dataset to load. Default is "train".

        Returns:
            If a list is passed, returns a list containing loaded datasets.
            If a string is passed, returns a single loaded dataset.

        Raises:
            ValueError: If the input format for 'req' is invalid.

        Note:
            The method utilizes the Hugging Face `datasets` library to download datasets based on the provided names.
            If an error occurs during dataset download, the method logs the error using the Python logging module.
            It returns `None` in case of an error.

        Example:
            >>> tsdae = TSDAE()
            >>> dataset_names = ['imdb', 'ag_news']
            >>> loaded_datasets = tsdae.load_dataset_from_hf(dataset_names, split="train")
            >>> for dataset in loaded_datasets:
            ...     print(len(dataset['train']))
        """
        try:
            if isinstance(dataset, list):
                data = []

                for element in dataset:
                    loaded_dataset = load_dataset(
                        element, split=split, streaming=streaming
                    )

                    data.append(loaded_dataset)

            elif isinstance(dataset, str):
                data = load_dataset(dataset, split=split, streaming=streaming)

            return data

        except Exception as e:
            self.logger.error(f"An error occurred while downloading datasets: {str(e)}")

            return None

    def load_dataset_from_dict(self, data: dict) -> list[Dataset] | Dataset:
        """
        Loads datasets from a dictionary using the Hugging Face datasets library.

        Args:
            data: A dictionary containing dataset names as keys and their corresponding configurations as values.

        Returns:
            Either a list of loaded datasets or a single loaded dataset.

        Raises:
            ValueError: If 'data_dict' is not a valid dictionary.

        Note:
            This method utilizes the Hugging Face `datasets` library to load datasets based on the provided dictionary.
            If an error occurs during dataset loading, it raises a ValueError.

        Example:
            >>> tsdae = TSDAE()
            >>> data_config = {'imdb': {'split': 'train'}, 'ag_news': {'split': 'train'}}
            >>> loaded_datasets = tsdae.load_dataset_from_dict(data_config)
            >>> for dataset in loaded_datasets:
            ...     print(len(dataset))
        """
        try:
            # Validate input type
            if not isinstance(data, dict):
                raise InvalidDatasetTypeError("'data_dict' must be a dictionary.")

            if len(data) == 1:
                dataset = Dataset.from_dict(data)

                return dataset

            elif len(data) > 1:
                # Load multiple datasets and return as a list
                datasets_list = [Dataset.from_dict(element) for element in data]

                return datasets_list

            else:
                raise InvalidDatasetTypeError(
                    "No dataset names provided in 'data_dict'."
                )

        except Exception as e:
            raise DatasetLoadError(f"Error in load_dataset_from_dict: {str(e)}") from e

    def shuffle_dataset(
        self,
        dataset: list[Dataset] | Dataset | IterableDataset,
        seed: int | None = 42,
    ) -> list[Dataset] | Dataset | IterableDataset:
        """
        Shuffles the dataset within a list or a single dataset and returns the shuffled result.

        Args:
            dataset: Either a list containing datasets or a single dataset.
            seed: The seed value for shuffling. Default is 42.

        Returns:
            Either a new list containing shuffled datasets or a single shuffled dataset.

        Raises:
            ValueError: If 'datasets_list' is not a list or instance of Dataset.
        """
        try:
            # Validate input type
            if not isinstance(dataset, (list, Dataset, IterableDataset)):
                raise InvalidDatasetTypeError(
                    "'datasets_list' must be a list or an instance of Dataset."
                )

            if isinstance(dataset, list):
                # Shuffling a list of datasets
                shuffled_datasets_list = [
                    element.shuffle(seed=seed) for element in dataset
                ]

                return shuffled_datasets_list

            elif isinstance(dataset, (Dataset, IterableDataset)):
                # Shuffling a single dataset outside a list
                shuffled_dataset = dataset.shuffle(seed=seed)

                return shuffled_dataset

        except Exception as e:
            self.logger.error(f"Error in shuffle_datasets: {str(e)}")
            return dataset

    def splitter_config(self, regex: str | Pattern | None = None) -> Pattern:
        """
        Configures and returns a compiled regular expression pattern for text splitting.

        Args:
            regex: A compiled regular expression pattern for text splitting. Default is a pattern matching period followed by optional whitespace and newline.

        Returns:
            A compiled regular expression pattern used for text splitting.

        Raises:
            ValueError: If 'regex' is not a valid regular expression.

        Note:
            This method compiles the regular expression pattern for text splitting based on the provided 'regex'.
            If an error occurs during compilation, it raises a ValueError.

        Example:
            >>> tsdae = TSDAE()
            >>> tsdae.splitter_config()
            <re.Pattern object at 0x...>
        """
        try:
            if regex is None:
                regex = r"\.\s?\n?"  # noqa: W605
            splitter = re.compile(regex) if isinstance(regex, str) else regex

            return splitter

        except re.error as e:
            raise InvalidRegexPatternError(
                f"Error in splitter_config: Invalid regular expression - {str(e)}"
            ) from e

    def sample_adjustment(self, population: list[str], sample_size: int) -> list[str]:
        """
        Randomly samples elements from a population while handling scenarios where
        the sample size exceeds the population size.

        Args:
            population: The list of elements to sample from.
            sample_size: The number of elements to sample.

        Returns:
            A randomly sampled subset of the population.

        Note:
            This function ensures that the sample size does not exceed the population size,
            preventing errors. It uses random.sample internally for efficient random sampling.

        Example:
            >>> population_data = ['A', 'B', 'C', 'D', 'E']
            >>> desired_sample_size = 7
            >>> result_sample = custom_sample(population_data, desired_sample_size)
            >>> print(result_sample)
        """
        adjusted_sample_size = min(sample_size, len(population))

        return random.sample(population, adjusted_sample_size)

    def _extract_sentences_from_row(
        self, row: Mapping[str, Any], column: str, splitter: Pattern, num_chars: int
    ) -> list[str]:
        """
        Extracts and filters sentences from a dataset row.

        Args:
            row: A single row from the dataset.
            column: The column name containing the text.
            splitter: Pattern for splitting text into sentences.
            num_chars: Minimum characters required for a sentence.

        Returns:
            Filtered sentences from the row.
        """
        new_sentences = splitter.split(row[column])
        return [line for line in new_sentences if len(line) > num_chars]

    def _process_dataset_element(
        self,
        element: Dataset | IterableDataset,
        column: str,
        splitter: Pattern,
        num_chars: int,
    ) -> list[str]:
        """
        Processes a single dataset element to extract sentences.

        Args:
            element: A dataset element to process.
            column: The column name containing the text.
            splitter: Pattern for splitting text into sentences.
            num_chars: Minimum characters required for a sentence.

        Returns:
            All extracted sentences from the element.
        """
        element_sentences = []
        for row in element:
            # Type cast: dataset rows are dict-like mappings
            sentences = self._extract_sentences_from_row(
                cast(Mapping[str, Any], row),
                column,
                splitter,
                num_chars,
            )
            element_sentences.extend(sentences)
        return element_sentences

    def constrain_dataset(
        self,
        shuffled_dataset: list[Dataset] | Dataset | IterableDataset,
        column: str,
        splitter: Pattern,
        num_to_keep: int = 100000,
        total_sentences: int = 100000,
        num_chars: int = 40,
    ) -> list[str]:
        """
        Processes shuffled datasets to constrain the number of sentences in a dataset.

        Args:
            shuffled_dataset: Either a list of shuffled datasets or a single dataset.
            column: The name of the column containing sentences in the dataset.
            splitter: A compiled regular expression pattern used for text splitting.
            num_to_keep: The number of sentences to retain per dataset element. Default is 100000.
            total_sentences: The total number of sentences to include in the constrained dataset. Default is 100000.
            num_chars: The minimum number of characters a line must have to be included in the dataset. Default is 40.

        Returns:
            A constrained dataset containing a specified number of sentences.

        Raises:
            ValueError: If 'shuffled_dataset' is not a list or dictionary.

        Note:
            This method processes shuffled datasets to constrain the number of sentences.
            It works with both a list of datasets and a single dataset outside a list.
        """
        try:
            # Validate input type
            if not isinstance(shuffled_dataset, (list, Dataset, IterableDataset)):
                raise InvalidDatasetTypeError(
                    "'shuffled_datasets_list' must be a list or Dataset."
                )

            all_sentences = []

            if isinstance(shuffled_dataset, list):
                for element in shuffled_dataset:
                    element_sentences = self._process_dataset_element(
                        element, column, splitter, num_chars
                    )
                    if len(element_sentences) <= num_to_keep:
                        all_sentences.extend(element_sentences)
                population = all_sentences
            else:
                element_sentences = self._process_dataset_element(
                    shuffled_dataset, column, splitter, num_chars
                )
                if len(element_sentences) <= num_to_keep:
                    all_sentences.extend(element_sentences)
                population = all_sentences if all_sentences else element_sentences

            constrained_dataset = self.sample_adjustment(
                population=population, sample_size=num_to_keep
            )

            return constrained_dataset

        except Exception as e:
            raise DatasetConstraintError(f"Error in constrain_dataset: {str(e)}") from e

    def create_denoising_loader(
        self,
        sentences: list[str],
        batch_size: int | None = 4,
        num_workers: int | None = 0,
    ) -> DataLoader:
        """
        Creates a DataLoader with noise functionality for a DenoisingAutoEncoderDataset.

        Args:
            sentences: List of sentences to be used in the dataset.
            batch_size: The batch size for the DataLoader. Default is 4.
            num_workers: Number of worker processes for data loading. Default is 0 (single-process data loading).
                Set to a positive integer to enable multiprocessing data loading.

        Returns:
            DataLoader object containing the DenoisingAutoEncoderDataset with specified noise functionality.

        Raises:
            ValueError: If 'sentences' is not a list, if 'batch_size' is not a positive integer,
                or if 'num_workers' is not a non-negative integer.

        Note:
            This method creates a DataLoader with noise functionality for a DenoisingAutoEncoderDataset.
            The DenoisingAutoEncoderDataset is initialized with the provided list of sentences.
            The DataLoader is configured with the specified batch size, shuffle, drop_last, and num_workers parameters.
        """
        try:
            # Validate input types
            if not isinstance(sentences, list):
                raise InvalidSentencesInputError("Input 'sentences' must be a list.")

            if not isinstance(batch_size, int) or batch_size <= 0:
                raise InvalidBatchSizeError(
                    "Input 'batch_size' must be a positive integer."
                )

            if not isinstance(num_workers, int) or num_workers < 0:
                raise InvalidNumWorkersError(
                    "Input 'num_workers' must be a non-negative integer."
                )

            # Create DenoisingAutoEncoderDataset
            train_data = DenoisingAutoEncoderDataset(sentences)

            # Create DataLoader
            loader = DataLoader(
                train_data,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=num_workers,
            )

            return loader

        except Exception as e:
            raise DataLoaderCreationError(
                f"Error in create_denoising_loader: {str(e)}"
            ) from e

    def create_sentence_transformer_model(
        self,
        model_name: str = "bert-base-multilingual-uncased",
        pooling_method: str | None = "cls",
    ) -> SentenceTransformer:
        """
        Creates a SentenceTransformer model using a specified transformer and pooling method.

        Args:
            model_name: The name of the transformer model to use. Default is "bert-base-multilingual-uncased".
            pooling_method: The pooling method to use. Default is "cls".

        Returns:
            SentenceTransformer model initialized with the specified transformer and pooling methods.

        Raises:
            ValueError: If 'model_name' is not a string.

        Note:
            This method creates a SentenceTransformer model by initializing a transformer (BERT by default)
            and a pooling layer with the specified model name and pooling method.
        """
        try:
            # Validate input type
            if not isinstance(model_name, str):
                raise InvalidModelNameError("Input 'model_name' must be a string.")

            # Create SentenceTransformer model
            transformer = models.Transformer(model_name)

            pooling = models.Pooling(
                transformer.get_word_embedding_dimension(), pooling_method
            )

            model = SentenceTransformer(modules=[transformer, pooling])

            return model

        except Exception as e:
            raise ModelCreationError(
                f"Error in create_sentence_transformer_model: {str(e)}"
            ) from e

    def create_denoising_autoencoder_loss(
        self, model: SentenceTransformer, tie_encoder_decoder: bool = True
    ) -> DenoisingAutoEncoderLoss:
        """
        Creates a DenoisingAutoEncoderLoss object for a given model.

        Args:
            model: SentenceTransformer model initialized with the specified transformer and pooling methods.
            tie_encoder_decoder: Determines whether to tie encoder and decoder parameters. Default is True.

        Returns:
            DenoisingAutoEncoderLoss object initialized for the given model.

        Raises:
            ValueError: If 'model' is not provided or if 'tie_encoder_decoder' is not a boolean.

        Note:
            This method creates a DenoisingAutoEncoderLoss object for a given model.
            The loss is computed based on the provided model and the optional parameter to tie encoder and decoder.
        """
        try:
            # Validate input types
            if model is None:
                raise InvalidModelInputError("Input 'model' must be provided.")

            if not isinstance(tie_encoder_decoder, bool):
                raise InvalidTieEncoderDecoderError(
                    "Input 'tie_encoder_decoder' must be a boolean."
                )

            # Create DenoisingAutoEncoderLoss object
            loss = DenoisingAutoEncoderLoss(
                model, tie_encoder_decoder=tie_encoder_decoder
            )

            return loss

        except Exception as e:
            raise LossCreationError(
                f"Error in create_denoising_autoencoder_loss: {str(e)}"
            ) from e

    def train_model(
        self,
        model: SentenceTransformer,
        output_path: str,
        loader: DataLoader,
        loss: DenoisingAutoEncoderLoss,
        epochs: int = 1,
        weight_decay: float = 0,
        scheduler: str = "constantlr",
        optimizer_params: dict[str, Any] | None = None,
        show_progress_bar: bool = True,
    ) -> SentenceTransformer:
        """
        Trains the specified model using the provided DataLoader and loss function.

        Args:
            model: The model to be trained.
            output_path: The file path to save the model.
            loader: DataLoader containing the training data.
            loss: Loss function used for training.
            epochs: Number of epochs for training. Default is 1.
            weight_decay: Weight decay value for regularization. Default is 0.
            scheduler: Scheduler type for adjusting learning rate. Default is 'constantlr'.
            optimizer_params: Parameters for the optimizer. Default is {'lr': 3e-5}.
            show_progress_bar: Whether to display a progress bar during training. Default is True.

        Returns:
            The trained model.

        Raises:
            ValueError: If 'model', 'path', 'loader', or 'loss' is not provided.

        Note:
            This method trains the specified model using the provided DataLoader and loss function.
            It supports customization of training parameters such as epochs, weight decay, scheduler type, and optimizer parameters.
        """
        try:
            if optimizer_params is None:
                optimizer_params = {"lr": 3e-5}
            # Train the model
            model.fit(
                train_objectives=[(loader, loss)],
                epochs=epochs,
                weight_decay=weight_decay,
                scheduler=scheduler,
                optimizer_params=optimizer_params,
                show_progress_bar=show_progress_bar,
            )

            # Save the trained model
            model.save(output_path)

            return model

        except ValueError as ve:
            raise ValidationError(f"ValueError in train: {str(ve)}") from ve

        except TypeError as te:
            raise TypeMismatchError(f"TypeError in train: {str(te)}") from te

        except Exception as e:
            raise TrainingError(f"Error in train: {str(e)}") from e

    def train(
        self,
        train_dataset: Dataset | IterableDataset,
        model_name: str,
        column: str,
        output_path: str,
        epochs: int = 1,
        weight_decay: float = 0,
        scheduler: str = "constantlr",
        optimizer_params: dict[str, Any] | None = None,
        show_progress_bar: bool = True,
        num_to_keep: int = 100000,
        total_sentences: int = 100000,
        num_chars: int = 40,
    ) -> SentenceTransformer:
        """
        Trains the specified model using the provided DataLoader and loss function.

        Args:
            train_dataset: The dataset to be used for training.
            model_name: The name of the model to be created.
            column: The name of the column containing sentences in the dataset.
            output_path: The file path to save the trained model.
            epochs: Number of epochs for training. Default is 1.
            weight_decay: Weight decay value for regularization. Default is 0.
            scheduler: Scheduler type for adjusting learning rate. Default is 'constantlr'.
            optimizer_params: Parameters for the optimizer. Default is {'lr': 3e-5}.
            show_progress_bar: Whether to display a progress bar during training. Default is True.
            num_to_keep: The number of sentences to retain per dataset element. Default is 15000.
            total_sentences: The total number of sentences to include in the constrained dataset. Default is 100000.
            num_chars: The minimum number of characters a line must have to be included in the dataset. Default is 40.

        Returns:
            The trained model.

        Raises:
            ValueError: If 'model_name', 'output_path', 'loader', or 'loss' is not provided.
            TypeError: If 'train_dataset' is not an instance of Dataset.

        Note:
            This method trains the specified model using the provided DataLoader and loss function.
            It supports customization of training parameters such as epochs, weight decay, scheduler type, and optimizer parameters.
        """
        try:
            # Validate required parameters
            if not model_name or not output_path or not column:
                raise ValidationError(
                    "Required parameters 'model_name', 'output_path', 'column' not provided."
                )

            # Validate dataset type
            if not isinstance(train_dataset, (Dataset, IterableDataset)):
                raise TypeMismatchError(
                    "'train_dataset' must be an instance of Dataset."
                )

            # Shuffle datasets
            shuffled_dataset = self.shuffle_dataset(
                dataset=train_dataset,
            )

            # Configure splitter
            splitter = self.splitter_config()

            # Constrain dataset
            constrained_data = self.constrain_dataset(
                shuffled_dataset=shuffled_dataset,
                column=column,
                splitter=splitter,
                num_to_keep=num_to_keep,
                total_sentences=total_sentences,
                num_chars=num_chars,
            )

            # Create DataLoader
            loader = self.create_denoising_loader(sentences=constrained_data)

            # Create model
            model = self.create_sentence_transformer_model(model_name=model_name)

            # Create loss function
            loss = self.create_denoising_autoencoder_loss(
                model=model, tie_encoder_decoder=True
            )

            if optimizer_params is None:
                optimizer_params = {"lr": 3e-5}

            # Train the model
            model = self.train_model(
                model=model,
                output_path=output_path,
                loader=loader,
                loss=loss,
                epochs=epochs,
                weight_decay=weight_decay,
                scheduler=scheduler,
                optimizer_params=optimizer_params,
                show_progress_bar=show_progress_bar,
            )

            return model

        except ValueError as ve:
            raise ValidationError(f"ValueError in train: {str(ve)}") from ve

        except TypeError as te:
            raise TypeMismatchError(f"TypeError in train: {str(te)}") from te

        except Exception as e:
            raise TrainingError(f"Error in train: {str(e)}") from e
