import os
import logging
import boto3
import tarfile
import zipfile
import json
import pandas as pd
import numpy as np

from tqdm import tqdm
from botocore.handlers import disable_signing

log = logging.getLogger(__name__)


class INaturalist2017Preparator:
    """
    Handles downloading, extracting, and preparing the iNaturalist 2017 dataset.

    Automates the process of downloading and extracting specific splits of the dataset
    from AWS S3, managing these processes through a simplified interface.

    Attributes:
        _dataset_path (str): Local file system path where the dataset will be stored.
    """

    SPLITS = {
        "train_val": {
            "images_file_name": "train_val_images.tar.gz",
            "annotations_file_name": "train_val2017.zip",
            "images_bytes": 199332058343
        }
    }

    def __init__(self,
                 dataset_path,
                 min_images_per_category,
                 fully_outer_supercategories,
                 fully_inner_supercategories,
                 outer_classes_size,
                 inner_val_size
                 ):
        """
        Initialize the INaturalist2017 object with the specified dataset path.

        Ensures the dataset is available at the specified path, downloading and extracting if necessary.

        Args:
            dataset_path (str): Local path where the dataset will be saved.
            min_images_per_category (int): The minimum number of images required for a category to be included.
            fully_outer_supercategories (list of str): List of supercategories that should be entirely classified as 'outer'. All categories under these supercategories will be assigned to an 'outer' split.
            fully_inner_supercategories (list of str): List of supercategories that should be entirely classified as 'inner'. All categories under these supercategories will be assigned to an 'inner' split.
            outer_classes_size (float): A decimal representing the proportion of categories within a non-fully classified supercategory to be randomly assigned to the 'outer' split. The rest will be assigned to the 'inner' split.
            inner_val_size (float): A decimal indicating the proportion of data within 'inner' categories to be used as validation data ('inner_val'). The rest of the data in 'inner' categories will be used for training ('inner_train').
        """
        self._dataset_path = dataset_path
        self._min_images_per_category = min_images_per_category
        self._fully_outer_supercategories = fully_outer_supercategories
        self._fully_inner_supercategories = fully_inner_supercategories
        self._outer_classes_size = outer_classes_size
        self._inner_val_size = inner_val_size

        self._ensure_dataset()
        self._dataframe = self._process_annotations_and_make_splits()

        self._dataframe.to_csv(os.path.join(dataset_path, "prepared_data.csv"))

    def _download_dataset_split(self, split_name):
        """
        Download, extract, and clean up a specific dataset split and its annotations.
        """
        # Handling dataset file
        file_name = self.SPLITS[split_name]["images_file_name"]
        file_path = os.path.join(self._dataset_path, file_name)
        self._download_file(file_name, file_path)
        self._extract_file(file_path, self.SPLITS[split_name]["images_bytes"])
        self._cleanup(file_path)

        # Handling annotations file
        annotations_file_name = self.SPLITS[split_name]["annotations_file_name"]
        annotations_file_path = os.path.join(self._dataset_path, annotations_file_name)
        self._download_file(annotations_file_name, annotations_file_path)
        self._extract_file(annotations_file_path)
        self._cleanup(annotations_file_path)

    @staticmethod
    def _download_file(file_name, file_path):
        """
        Download a file from AWS S3.

        Args:
            file_name (str): Name of the file to download.
            file_path (str): Path to save the downloaded file.
        """
        s3 = boto3.client('s3')
        s3.meta.events.register('choose-signer.s3.*', disable_signing)
        bucket_name = 'ml-inat-competition-datasets'
        object_name = f'2017/{file_name}'

        log.info(f"Downloading {file_name}...")
        try:
            response = s3.head_object(Bucket=bucket_name, Key=object_name)
            file_size = response['ContentLength']

            with tqdm(total=file_size, unit='B', unit_scale=True, desc=file_name) as progress:
                s3.download_file(Bucket=bucket_name, Key=object_name, Filename=file_path,
                                 Callback=lambda chunk: progress.update(chunk))
            log.info(f"{file_name} download successful")
        except Exception as e:
            log.error(f"Error downloading {file_name}: {e}")
            raise

    @staticmethod
    def _extract_file(file_path, total_bytes=None):
        """
        Extract a file (either tar.gz or zip) from the given file path.

        Args:
            file_path (str): Path of the file to be extracted.
            total_bytes (int, optional): Total size in bytes for progress tracking (for tar.gz files).
        """
        log.info(f"Extracting {os.path.basename(file_path)}...")
        try:
            if file_path.endswith('.tar.gz'):
                with tarfile.open(file_path, 'r:gz') as tar, \
                        tqdm(unit='B', unit_scale=True, desc="Extracting", total=total_bytes) as progress:
                    for member in tar:
                        tar.extract(member, path=file_path)
                        progress.update(member.size)
            elif file_path.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(file_path))
            log.info(f"Extraction of {os.path.basename(file_path)} successful")
        except Exception as e:
            log.error(f"Error extracting {file_path}: {e}")
            raise

    @staticmethod
    def _cleanup(file_path):
        """
        Remove a specified file from the file system after successful extraction.

        Args:
            file_path (str): Path of the file to be removed.
        """
        try:
            os.remove(file_path)
            log.info(f"Removed file {file_path}")
        except Exception as e:
            log.error(f"Error removing file {file_path}: {e}")

    def _ensure_dataset(self):
        """
        Ensure the dataset is available in the specified directory.

        Checks if the dataset directory exists and triggers the download
        and extraction process if not found.
        """
        if not os.path.exists(self._dataset_path):
            log.info("Dataset path does not exist. Creating directories and downloading dataset.")
            os.makedirs(self._dataset_path)
            self._download_dataset_split('train_val')
        else:
            log.info("Dataset path exists. No download required.")

    def _load_and_merge_splits(self):
        """
        Loads training and validation splits from JSON files and merges them.

        This function reads the 'train2017.json' and 'val2017.json' files to extract
        image, category, and annotation data. It then creates Pandas DataFrames for each
        type of data and merges training and validation datasets where appropriate.

        Returns:
            tuple: A tuple containing merged DataFrames for images, categories, and annotations.
        """
        with open(os.path.join(self._dataset_path, 'train2017.json'), 'r') as file:
            train_data = json.load(file)
        with open(os.path.join(self._dataset_path, 'val2017.json'), 'r') as file:
            val_data = json.load(file)

        # Create DataFrames for images, categories, and annotations
        train_df_images = pd.DataFrame(train_data['images'])
        val_df_images = pd.DataFrame(val_data['images'])

        train_df_annotations = pd.DataFrame(train_data['annotations'])
        val_df_annotations = pd.DataFrame(val_data['annotations'])

        train_df_categories = pd.DataFrame(train_data['categories'])

        train_df_images["source"] = "original_train"
        val_df_images["source"] = "original_val"

        merged_images = pd.concat([train_df_images, val_df_images], ignore_index=True)
        merged_categories = train_df_categories  # train_df_categories.equals(val_df_categories)
        merged_annotations = pd.concat([train_df_annotations, val_df_annotations], ignore_index=True)

        return merged_images, merged_categories, merged_annotations

    def _process_annotations(self):
        """
        Loads data from a JSON file and processes it to create a DataFrame with image and category details.

        The function filters categories with more than 'min_images_per_category' images and merges image and category information.

        Returns:
        pd.DataFrame: A DataFrame with columns ['image_id', 'category_id', 'image_path', 'category', 'supercategory'].
        """
        df_images, df_categories, df_annotations = self._load_and_merge_splits()

        # Count the number of images per category in df_annotations
        image_count_per_category = df_annotations['category_id'].value_counts().reset_index()
        image_count_per_category.columns = ['id', 'num_images']

        # Merge this count with df_categories
        df_categories = pd.merge(df_categories, image_count_per_category, on='id', how='left')

        # Filter df_categories for those with more than min_images_per_category images
        filtered_categories = df_categories[df_categories['num_images'] > self._min_images_per_category]

        # Merge filtered_categories with df_annotations
        df_merged = pd.merge(filtered_categories, df_annotations, left_on='id', right_on='category_id')

        # Merge with df_images to get image details
        df_final = pd.merge(df_merged, df_images, left_on='image_id', right_on='id')

        # Select and rename the required columns
        df_final = df_final[['image_id', 'category_id', 'file_name', 'name', 'supercategory', 'source']]
        df_final.rename(columns={'file_name': 'image_path', 'name': 'category'}, inplace=True)

        return df_final

    def _process_annotations_and_make_splits(self):
        """
        Processes annotations and creates custom splits in the resulting DataFrame.

        This method first processes annotations to create a DataFrame and then applies custom logic to create data splits based on supercategories and categories.

        Returns:
            pd.DataFrame: A DataFrame with processed annotations and custom splits.
        """
        # Process annotations to create a DataFrame
        dataframe = self._process_annotations()

        # Apply custom logic to create data splits
        self._make_custom_splits(dataframe)

        return dataframe

    def _make_custom_splits(self, df):
        """
        Creates custom splits in the DataFrame based on predefined supercategories.

        The method first assigns 'outer' and 'inner' splits based on supercategories. Then, for remaining supercategories, it randomly assigns categories to 'outer' or 'inner' splits based on a predefined ratio. Finally, it assigns 'inner_train' and 'inner_val' splits for the 'inner' categories.

        Args:
            df (pd.DataFrame): The DataFrame in which to create the splits.
        """
        # Initialize the 'split' column to None
        df["split"] = None

        # Assign 'outer' and 'inner' splits based on supercategories
        outer_mask = df["supercategory"].isin(self._fully_outer_supercategories)
        inner_mask = df["supercategory"].isin(self._fully_inner_supercategories)
        df.loc[outer_mask, "split"] = "outer"
        df.loc[inner_mask, "split"] = "inner"

        # Process remaining supercategories
        remaining_supercategories = set(df.loc[~(outer_mask | inner_mask), "supercategory"].unique())
        for supercategory in remaining_supercategories:
            supercategory_mask = df["supercategory"] == supercategory
            categories = df.loc[supercategory_mask, "category"].unique()
            outer_count = np.ceil(len(categories) * self._outer_classes_size).astype(int)
            outer_categories = np.random.choice(categories, size=outer_count, replace=False)

            category_outer_mask = df["category"].isin(outer_categories)
            df.loc[category_outer_mask, "split"] = "outer"
            df.loc[supercategory_mask & ~category_outer_mask, "split"] = "inner"

        # Assign 'inner_train' and 'inner_val' splits for 'inner' categories
        inner_categories = df.loc[df["split"] == "inner", "category"].unique()
        for c in inner_categories:
            category_mask = (df["category"] == c)
            df.loc[category_mask, 'split'] = np.where(np.random.rand(category_mask.sum()) < self._inner_val_size,
                                                      'inner_train',
                                                      'inner_val')
