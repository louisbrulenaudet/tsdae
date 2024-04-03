# -*- coding: utf-8 -*-
# Copyright (c) Louis Brulé Naudet. All Rights Reserved.
# This software may be used and distributed according to the terms of License Agreement.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import ssl
import nltk
import random
import logging
from typing import Union, Optional
from datasets import concatenate_datasets, load_dataset, Dataset

from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, models
from sentence_transformers.losses import DenoisingAutoEncoderLoss


def create_unverified_https_context():
    """
    Create an unverified HTTPS context if available.

    This function checks for the availability of an unverified HTTPS context and sets it as the default HTTPS context
    if it's available. If not available, it maintains the default behavior.

    Raises
    ------
    RuntimeError
        If an error occurs during the creation of the unverified HTTPS context.

    Notes
    -----
    This functionality is primarily used to handle SSL certificate verification in HTTPS connections, allowing
    for unverified contexts when necessary.
    """
    try:
        _create_unverified_https_context = ssl._create_unverified_context
        ssl._create_default_https_context = _create_unverified_https_context

    except AttributeError as e:
        raise RuntimeError("Unable to create an unverified HTTPS context.") from e

try:
    create_unverified_https_context()
    nltk.download("punkt")

except LookupError as e:
    raise LookupError("Failed to download the 'punkt' corpus from NLTK.") from e


class TSDAE:
    """
    Learning sentence embeddings often requires a large amount of labeled data.
    However, for most tasks and domains, labeled data is seldom available and creating it is expensive.
    In this work, we present a new state-of-the-art unsupervised method based on pre-trained
    Transformers and Sequential Denoising Auto-Encoder (TSDAE) which outperforms previous approaches
    by up to 6.4 points. It can achieve up to 93.1% of the performance of in-domain supervised approaches.

    The model architecture of TSDAE is a modified encoder-decoder Transformer where the key
    and value of the cross-attention are both confined to the sentence embedding only.

    Examples
    --------
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
    def __init__(self):
        """
        Initializes an instance of TSDAE with enhanced logging for error handling.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This constructor initializes the instance of TSDAE with a sophisticated logging configuration to 
        facilitate comprehensive error handling. The logging level is set to ERROR, ensuring that 
        only critical messages and above are recorded. The logger instance is created with 
        the class name (__name__), providing clear identification in logs. Additionally, a 
        StreamHandler is added to direct log output to the console.

        Examples
        --------
        >>> instance = TSDAE()
        """
        logging.basicConfig(level=logging.ERROR)

        # Create a logger with the class name
        self.logger = logging.getLogger(__name__)

        # Add a StreamHandler to direct log output to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)


    def load_dataset_from_hf(
        self,
        dataset: Union[list, str],
        streaming: Optional[bool] = True,
        split: Optional[str] = "train"
        ) -> Union[list, Dataset, IterableDataset]:
        """
        Downloads dataset(s) specified either in a list or as a string and returns the loaded datasets.

        Parameters
        ----------
        dataset : Union[list, str]
            A list containing the names of datasets to be downloaded or a
            string representing a single dataset name.

        streaming : bool, optional
            Determines if datasets are streamed. Default is True.

        split : str, optional
            The split of the dataset to load. Default is "train".

        Returns
        -------
        datasets : Union[list, Dataset, IterableDataset]
            If a list is passed, returns a list containing loaded datasets.
            If a string is passed, returns a single loaded dataset.

        Raises
        ------
        ValueError
            If the input format for 'req' is invalid.

        Notes
        -----
        The method utilizes the Hugging Face `datasets` library to download datasets based on the provided names.
        If an error occurs during dataset download, the method logs the error using the Python logging module.
        It returns `None` in case of an error.

        Example
        -------
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
                    dataset = load_dataset(
                        element,
                        split=split,
                        streaming=streaming
                    )

                    data.append(dataset)

            elif isinstance(dataset, str):
                data = load_dataset(
                    dataset,
                    split=split,
                    streaming=streaming
                )

            else:
                raise ValueError(
                    "Invalid input format for 'dataset'. It should be a list or a string."
                )

            return data

        except Exception as e:
            self.logger.error(
                f"An error occurred while downloading datasets: {str(e)}"
            )

            return None


    def load_dataset_from_dict(
        self,
        data:dict
        ) -> Union[list, Dataset, IterableDataset]:
        """
        Loads datasets from a dictionary using the Hugging Face datasets library.

        Parameters
        ----------
        data : dict
            A dictionary containing dataset names as keys and their corresponding configurations as values.

        Returns
        -------
        datasets : Union[list, Dataset]
            Either a list of loaded datasets or a single loaded dataset.

        Raises
        ------
        ValueError
            If 'data_dict' is not a valid dictionary.

        Notes
        -----
        This method utilizes the Hugging Face `datasets` library to load datasets based on the provided dictionary.
        If an error occurs during dataset loading, it raises a ValueError.

        Example
        -------
        >>> tsdae = TSDAE()
        >>> data_config = {'imdb': {'split': 'train'}, 'ag_news': {'split': 'train'}}
        >>> loaded_datasets = tsdae.load_dataset_from_dict(data_config)
        >>> for dataset in loaded_datasets:
        ...     print(len(dataset))
        """
        try:
            # Validate input type
            if not isinstance(data, dict):
                raise ValueError("'data_dict' must be a dictionary.")

            if len(data) == 1:
                dataset = Dataset.from_dict(
                    data
                )

                return dataset

            elif len(data) > 1:
                # Load multiple datasets and return as a list
                datasets_list = [
                    Dataset.from_dict(
                        element
                    ) for element in data
                ]

                return datasets_list

            else:
                raise ValueError("No dataset names provided in 'data_dict'.")

        except Exception as e:
            raise ValueError(f"Error in load_dataset_from_dict: {str(e)}")


    def shuffle_dataset(
        self,
        dataset: Union[List[Dataset], Dataset, IterableDataset],
        seed:Optional[int] = 42
        ) -> Union[List[Dataset], Dataset, IterableDataset]:
        """
        Shuffles the dataset within a list or a single dataset and returns the shuffled result.

        Parameters
        ----------
        dataset : Union[List[Dataset], Dataset, IterableDataset]
            Either a list containing datasets or a single dataset.

        seed : int, optional
            The seed value for shuffling. Default is 42.

        Returns
        -------
        shuffled_datasets : Union[List[Dataset, IterableDataset], Dataset, IterableDataset]
            Either a new list containing shuffled datasets or a single shuffled dataset.

        Raises
        ------
        ValueError
            If 'datasets_list' is not a list or instance of Dataset.
        """
        try:
            # Validate input type
            if not (isinstance(dataset, list) or isinstance(dataset, (Dataset, IterableDataset))):
                raise ValueError("'datasets_list' must be a list or an instance of Dataset.")

            if isinstance(dataset, list):
                # Shuffling a list of datasets
                shuffled_datasets_list = [
                    element.shuffle(
                        seed=seed
                    ) for element in dataset
                ]

                return shuffled_datasets_list

            elif isinstance(dataset, (Dataset, IterableDataset)):
                # Shuffling a single dataset outside a list
                shuffled_dataset = dataset.shuffle(
                    seed=seed
                )

                return shuffled_dataset

        except Exception as e:
            self.logger.error(f"Error in shuffle_datasets: {str(e)}")
            return dataset


    def splitter_config(
        self,
        regex: Optional[Pattern]=re.compile(r'\.\s?\n?')
        ) -> Pattern:
        """
        Configures and returns a compiled regular expression pattern for text splitting.

        Parameters
        ----------
        regex : Optional[re.Pattern], optional
            A compiled regular expression pattern for text splitting. Default is `r'\.\s?\n?'`.

        Returns
        -------
        splitter : re.Pattern
            A compiled regular expression pattern used for text splitting.

        Raises
        ------
        ValueError
            If 'regex' is not a valid regular expression.

        Notes
        -----
        This method compiles the regular expression pattern for text splitting based on the provided 'regex'.
        If an error occurs during compilation, it raises a ValueError.

        Example
        -------
        >>> tsdae = TSDAE()
        >>> tsdae.splitter_config()
        <re.Pattern object at 0x...>
        """
        try:
            splitter = re.compile(
                regex
            )

            return splitter

        except re.error as e:
            raise Exception(f"Error in splitter_config: Invalid regular expression - {str(e)}")


    def sample_adjustment(
        self,
        population: List[str], 
        sample_size: int
        ) -> List[str]:
        """
        Randomly samples elements from a population while handling scenarios where
        the sample size exceeds the population size.

        Parameters
        ----------
        population : List[str]
            The list of elements to sample from.

        sample_size : int
            The number of elements to sample.

        Returns
        -------
        sampled_elements : List[str]
            A randomly sampled subset of the population.

        Notes
        -----
        This function ensures that the sample size does not exceed the population size,
        preventing errors. It uses random.sample internally for efficient random sampling.

        Examples
        --------
        >>> population_data = ['A', 'B', 'C', 'D', 'E']
        >>> desired_sample_size = 7
        >>> result_sample = custom_sample(population_data, desired_sample_size)
        >>> print(result_sample)
        """
        adjusted_sample_size = min(
            sample_size, 
            len(population)
        )
        
        return random.sample(
            population, 
            adjusted_sample_size
        )


    def constrain_dataset(
        self,
        shuffled_dataset:Union[List[Dataset], Dataset, IterableDataset],
        column:str,
        splitter:re.Pattern,
        num_to_keep:int=100000,
        total_sentences:int=100000,
        num_chars:Optional[int] = 40
        ) -> list:
        """
        Processes shuffled datasets to constrain the number of sentences in a dataset.

        Parameters
        ----------
        shuffled_dataset : Union[List[Dataset], Dataset, IterableDataset]
            Either a list of shuffled datasets or a single dataset.

        column : str
            The name of the column containing sentences in the dataset.

        splitter : Pattern
            A compiled regular expression pattern used for text splitting.

        num_to_keep : int, optional
            The number of sentences to retain per dataset element. Default is 100000.

        total_sentences : int, optional
            The total number of sentences to include in the constrained dataset. Default is 100000.

        num_chars : int, optional
            The minimum number of characters a line must have to be included in the dataset. Default is 40.

        Returns
        -------
        constrained_dataset : List[str]
            A constrained dataset containing a specified number of sentences.

        Raises
        ------
        ValueError
            If 'shuffled_dataset' is not a list or dictionary.

        Notes
        -----
        This method processes shuffled datasets to constrain the number of sentences.
        It works with both a list of datasets and a single dataset outside a list.
        """
        try:
            # Validate input type
            if not (isinstance(shuffled_dataset, list) or isinstance(shuffled_dataset, (Dataset, IterableDataset))):
                raise ValueError("'shuffled_datasets_list' must be a list or Dataset.")

            sentences = []

            if isinstance(shuffled_dataset, list):
                for element in shuffled_dataset:
                    element_sentences = []

                    for row in element:
                        new_sentences = splitter.split(
                            row[column]
                        )

                        new_sentences = [
                            line for line in new_sentences if len(line) > num_chars
                        ]

                        element_sentences.extend(new_sentences)

                    if len(element_sentences) <= num_to_keep:
                        sentences.extend(element_sentences)

            elif isinstance(shuffled_dataset, (Dataset, IterableDataset)):
                element_sentences = []

                for row in shuffled_dataset:
                    new_sentences = splitter.split(
                        row[column]
                    )

                    new_sentences = [
                        line for line in new_sentences if len(line) > num_chars
                    ]

                    element_sentences.extend(new_sentences)

                if len(element_sentences) <= num_to_keep:
                    sentences.extend(element_sentences)

            constrained_dataset = self.sample_adjustment(
                population=element_sentences, 
                sample_size=num_to_keep
            )

            return constrained_dataset

        except Exception as e:
            raise Exception(f"Error in constrain_dataset: {str(e)}")


    def create_denoising_loader(
        self,
        sentences: list,
        batch_size: Optional[int] = 4
        ) -> DataLoader:
        """
        Creates a DataLoader with noise functionality for a DenoisingAutoEncoderDataset.

        Parameters
        ----------
        sentences : list
            List of sentences to be used in the dataset.

        batch_size : int, optional
            The batch size for the DataLoader. Default is 4.

        Returns
        -------
        loader : DataLoader
            DataLoader object containing the DenoisingAutoEncoderDataset with specified noise functionality.

        Raises
        ------
        ValueError
            If 'sentences' is not a list or if 'batch_size' is not a positive integer.

        Notes
        -----
        This method creates a DataLoader with noise functionality for a DenoisingAutoEncoderDataset.
        The DenoisingAutoEncoderDataset is initialized with the provided list of sentences.
        The DataLoader is configured with the specified batch size, shuffle, and drop_last parameters.
        """
        try:
            # Validate input types
            if not isinstance(sentences, list):
                raise ValueError("Input 'sentences' must be a list.")

            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError("Input 'batch_size' must be a positive integer.")

            # Create DenoisingAutoEncoderDataset
            train_data = DenoisingAutoEncoderDataset(sentences)

            # Create DataLoader
            loader = DataLoader(
                train_data,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True
            )

            return loader

        except Exception as e:
            raise Exception(f"Error in create_denoising_loader: {str(e)}")


    def create_sentence_transformer_model(
        self,
        model_name:str="bert-base-multilingual-uncased",
        pooling_method:Optional[str]="cls"
        ) -> SentenceTransformer:
        """
        Creates a SentenceTransformer model using a specified transformer and pooling method.

        Parameters
        ----------
        model_name : str, optional
            The name of the transformer model to use. Default is "bert-base-multilingual-uncased".

        pooling_method : str, optional
            The pooling method to use. Default is "cls".

        Returns
        -------
        model : SentenceTransformer
            SentenceTransformer model initialized with the specified transformer and pooling methods.

        Raises
        ------
        ValueError
            If 'model_name' is not a string.

        Notes
        -----
        This method creates a SentenceTransformer model by initializing a transformer (BERT by default)
        and a pooling layer with the specified model name and pooling method.
        """
        try:
            # Validate input type
            if not isinstance(model_name, str):
                raise ValueError("Input 'model_name' must be a string.")

            # Create SentenceTransformer model
            transformer = models.Transformer(
                model_name
            )

            pooling = models.Pooling(
                transformer.get_word_embedding_dimension(),
                pooling_method
            )

            model = SentenceTransformer(
                modules=[
                    transformer,
                    pooling
                ]
            )

            return model

        except Exception as e:
            raise Exception(
                f"Error in create_sentence_transformer_model: {str(e)}"
            )


    def create_denoising_autoencoder_loss(
        self,
        model:SentenceTransformer,
        tie_encoder_decoder:Optional[bool]=True
        ) -> DenoisingAutoEncoderLoss:
        """
        Creates a DenoisingAutoEncoderLoss object for a given model.

        Parameters
        ----------
        model : SentenceTransformer
            SentenceTransformer model initialized with the specified transformer and pooling methods.

        tie_encoder_decoder : bool, optional
            Determines whether to tie encoder and decoder parameters. Default is True.

        Returns
        -------
        loss : DenoisingAutoEncoderLoss
            DenoisingAutoEncoderLoss object initialized for the given model.

        Raises
        ------
        ValueError
            If 'model' is not provided or if 'tie_encoder_decoder' is not a boolean.

        Notes
        -----
        This method creates a DenoisingAutoEncoderLoss object for a given model.
        The loss is computed based on the provided model and the optional parameter to tie encoder and decoder.
        """
        try:
            # Validate input types
            if model is None:
                raise ValueError("Input 'model' must be provided.")

            if not isinstance(tie_encoder_decoder, bool):
                raise ValueError("Input 'tie_encoder_decoder' must be a boolean.")

            # Create DenoisingAutoEncoderLoss object
            loss = DenoisingAutoEncoderLoss(
                model,
                tie_encoder_decoder=tie_encoder_decoder
            )

            return loss

        except Exception as e:
            raise Exception(
                f"Error in create_denoising_autoencoder_loss: {str(e)}"
            )


    def train_model(
        self,
        model:any,
        output_path:str,
        loader:DataLoader,
        loss:DenoisingAutoEncoderLoss,
        epochs:int=1,
        weight_decay:float=0,
        scheduler:str="constantlr",
        optimizer_params:dict={"lr": 3e-5},
        show_progress_bar:bool=True
        ) -> None:
        """
        Trains the specified model using the provided DataLoader and loss function.

        Parameters
        ----------
        model : any
            The model to be trained.

        output_path : str
            The file path to save the model.

        loader : DataLoader
            DataLoader containing the training data.

        loss : DenoisingAutoEncoderLoss
            Loss function used for training.

        epochs : int, optional
            Number of epochs for training. Default is 1.

        weight_decay : float, optional
            Weight decay value for regularization. Default is 0.

        scheduler : str, optional
            Scheduler type for adjusting learning rate. Default is 'constantlr'.

        optimizer_params : dict, optional
            Parameters for the optimizer. Default is {'lr': 3e-5}.

        show_progress_bar : bool, optional
            Whether to display a progress bar during training. Default is True.

        Returns
        -------
        trained_model : any
            The trained model.

        Raises
        ------
        ValueError
            If 'model', 'path', 'loader', or 'loss' is not provided.

        Notes
        -----
        This method trains the specified model using the provided DataLoader and loss function.
        It supports customization of training parameters such as epochs, weight decay, scheduler type, and optimizer parameters.
        """
        try:
            # Train the model
            model.fit(
                train_objectives=[
                    (loader, loss)
                ],
                epochs=epochs,
                weight_decay=weight_decay,
                scheduler=scheduler,
                optimizer_params=optimizer_params,
                show_progress_bar=show_progress_bar
            )

            # Save the trained model
            model.save(output_path)

            return model

        except ValueError as ve:
            raise ValueError(f"ValueError in train: {str(ve)}")

        except TypeError as te:
            raise TypeError(f"TypeError in train: {str(te)}")

        except Exception as e:
            raise Exception(f"Error in train: {str(e)}")


    def train(
        self,
        train_dataset: Union[Dataset, IterableDataset],
        model_name: str,
        column:str,
        output_path: str,
        epochs: int = 1,
        weight_decay: float = 0,
        scheduler: str = "constantlr",
        optimizer_params: dict = {'lr': 3e-5},
        show_progress_bar: bool = True,
        num_to_keep: int = 100000,
        total_sentences: int = 100000,
        num_chars: int = 40
    ) -> None:
        """
        Trains the specified model using the provided DataLoader and loss function.

        Parameters
        ----------
        train_dataset : [Dataset, IterableDataset]
            The dataset to be used for training.

        model_name : str
            The name of the model to be created.

        column : str
            The name of the column containing sentences in the dataset.

        output_path : str
            The file path to save the trained model.

        epochs : int, optional
            Number of epochs for training. Default is 1.

        weight_decay : float, optional
            Weight decay value for regularization. Default is 0.

        scheduler : str, optional
            Scheduler type for adjusting learning rate. Default is 'constantlr'.

        optimizer_params : dict, optional
            Parameters for the optimizer. Default is {'lr': 3e-5}.

        show_progress_bar : bool, optional
            Whether to display a progress bar during training. Default is True.

        num_to_keep : int, optional
            The number of sentences to retain per dataset element. Default is 15000.

        total_sentences : int, optional
            The total number of sentences to include in the constrained dataset. Default is 100000.

        num_chars : int, optional
            The minimum number of characters a line must have to be included in the dataset. Default is 40.

        Returns
        -------
        trained_model : any
            The trained model.

        Raises
        ------
        ValueError
            If 'model_name', 'output_path', 'loader', or 'loss' is not provided.

        TypeError
            If 'train_dataset' is not an instance of Dataset.

        Notes
        -----
        This method trains the specified model using the provided DataLoader and loss function.
        It supports customization of training parameters such as epochs, weight decay, scheduler type, and optimizer parameters.
        """
        try:
            # Validate required parameters
            if not model_name or not output_path or not column:
                raise ValueError("Required parameters 'model_name', 'output_path', 'column' not provided.")

            # Validate dataset type
            if not isinstance(train_dataset, (Dataset, IterableDataset)):
                raise TypeError("'train_dataset' must be an instance of Dataset.")

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
                num_chars=num_chars
            )

            # Create DataLoader
            loader = self.create_denoising_loader(
                sentences=constrained_data
            )

            # Create model
            model = self.create_sentence_transformer_model(
                model_name=model_name
            )

            # Create loss function
            loss = self.create_denoising_autoencoder_loss(
                model=model,
                tie_encoder_decoder=True
            )

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
                show_progress_bar=show_progress_bar
            )

            return model

        except ValueError as ve:
            raise ValueError(f"ValueError in train: {str(ve)}")
            return None

        except TypeError as te:
            raise TypeError(f"TypeError in train: {str(te)}")
            return None

        except Exception as e:
            raise Exception(f"Error in train: {str(e)}")
            return None
