import copy
import numpy as np
import os
import random
import torch

from collections import Counter
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torch.utils.data import (
    Dataset,
    Subset,
    DataLoader,
    SequentialSampler,
    WeightedRandomSampler,
    random_split,
)


def set_seed(seed: int = 666):
    """
    Sets the random seed across Python, NumPy, and PyTorch to ensure reproducible results.

    Args:
        seed (int, optional): The seed value to use. Defaults to 666.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_metrics(metrics_dict):
    """
    Returns new dictionary with the same keys and Tensor metric values converted to Python floats.

    Args:
        metrics_dict (dict): Dictionary where keys are metric names and values are lists
            containing numbers or PyTorch tensors.

    Example:
        Input: {'loss': [tensor(0.5), tensor(0.3)], 'acc': [0.8, tensor(0.9)]}
        Output: {'loss': [0.5, 0.3], 'acc': [0.8, 0.9]}
    """

    cleaned = {}

    for key, values in metrics_dict.items():
        cleaned_values = []
        for v in values:
            if isinstance(v, torch.Tensor):
                cleaned_values.append(v.item())
            else:
                cleaned_values.append(float(v))
        cleaned[key] = cleaned_values

    return cleaned


def reorder_and_merge_classes(class_map, y_true, y_pred, y_prob, to_merge):
    """
    Reorders class indices and merges specified classes into a single 'Other' class.

    Args:
        class_map (dict): Mapping from class names to original indices.
        y_true (np.ndarray): Array of true labels with original indices.
        y_pred (np.ndarray): Array of predicted labels with original indices.
        y_prob (np.ndarray): Array of predicted probabilities of shape (n_samples, n_original_classes).
        to_merge (list): List of class names to merge into 'Other'.

    Returns:
        tuple:
            - new_class_map (dict): Mapping from class names (with 'Other') to new indices.
            - new_y_true (np.ndarray): True labels remapped to new indices.
            - new_y_pred (np.ndarray): Predicted labels remapped to new indices.
            - new_y_prob (np.ndarray): Probability array reshaped to (n_samples, n_new_classes),
              with probabilities of merged classes combined and normalized.

    Notes:
        - Probability vectors are summed across merged classes and renormalized to sum to 1, to avoid
          floating point error downstream.
    """

    class_map_rev = {v: k for k, v in class_map.items()}  # Inverse class map

    # Create new class map with main classes + other
    to_keep = sorted([cls for cls in class_map if cls not in to_merge])
    new_class_map = {cls: idx for idx, cls in enumerate(to_keep)}

    other_index = len(to_keep)
    new_class_map["Other"] = other_index

    # Map old indices to new indices
    orig_max_index = max(class_map.values())
    orig_to_new_map = np.zeros(orig_max_index + 1, dtype=int)

    for orig_idx, cls in class_map_rev.items():
        if cls in to_merge:
            orig_to_new_map[orig_idx] = other_index
        else:
            orig_to_new_map[orig_idx] = new_class_map[cls]

    # Map labels
    new_y_true = orig_to_new_map[y_true]
    new_y_pred = orig_to_new_map[y_pred]

    # Map prediction probabilities
    n_samples = y_prob.shape[0]
    new_n_classes = len(new_class_map)
    new_y_prob = np.zeros((n_samples, new_n_classes))

    for orig_idx, cls in class_map_rev.items():
        new_idx = new_class_map["Other"] if cls in to_merge else new_class_map[cls]
        new_y_prob[:, new_idx] += y_prob[:, orig_idx]

    # Need to normalize due to floating-point precision
    y_prob_sums = new_y_prob.sum(axis=1, keepdims=True)
    new_y_prob = new_y_prob / y_prob_sums

    return new_class_map, new_y_true, new_y_pred, new_y_prob


class ImageDataset(Dataset):
    """
    A custom PyTorch Dataset for loading and preprocessing image data from a directory
    where each subfolder represents a class.

    This class handles:
    - Class-wise and random sampling with optional size limits
    - Optional image transforms for data augmentation
    - Preprocessing and setup for imbalanced class handling

    Args:
        data_directory (str): Path to the root dataset directory. Each subdirectory should represent a class or another subdirectory to check.
        data_subdirectories (list of str, optional): Subdirectories with additional images, each sub-subdirectory should represent a class.
        class_names (list, optional): List of class names to include. If None, all subdirectories are included.
        class_sizes (list, optional): Number of samples to include per class. If None, uses `max_class_size` for all.
        class_ids (list, optional): Numeric ID for each class (aligned with `class_names`).
        max_class_size (int, optional): Default maximum number of samples to draw per class. Defaults to 10,000.
        image_resolution (int, optional): Final size (height and width) to resize images to. Defaults to 28.
        image_transforms (callable, optional): Image transformations (e.g., data augmentations) to apply. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to 666.

    Attributes:
        data_directory (str): Path to the dataset root directory.
        data_subdirectories (list of str): Subdirectories with additional images.
        seed (int): Random seed used for sampling.
        class_names (list): Sorted list of class names included in the dataset.
        class_sizes (torch.Tensor): Tensor of the actual sampled size per class.
        class_ids (list): Numeric ID for each class (aligned with `class_names`).
        image_paths (list): List of file paths to all sampled images.
        labels (list): List of numeric class IDs corresponding to each image.
        image_resolution (int): Size to which each image is resized.
        image_transforms (callable or None): Image transformations applied during training or inference.
    """

    def __init__(
        self,
        data_directory,
        data_subdirectories: list = None,
        class_names: list = None,
        class_sizes: list = None,
        class_ids: list = None,
        max_class_size: int = 10000,
        image_resolution: int = 28,
        image_transforms=None,
        seed: int = 666,
        aug_transforms=None,
        ignore_classes=None,
    ):
        self.data_directory = data_directory
        self.data_subdirectories = [""]
        self.seed = seed
        self.ignore_classes = set(ignore_classes)
        self.aug_transforms = aug_transforms

        set_seed(seed)

        # Additional subdirectories to check
        if data_subdirectories is not None:
            self.data_subdirectories.extend(data_subdirectories)

        # Specify subset of classes to consider; all classes considered if None
        if class_names is None:
            class_names = sorted(os.listdir(self.data_directory))

        # Specify initial number of samples to consider per class; max if None
        if class_sizes is None:
            class_sizes = [max_class_size] * len(class_names)

        # Specify numeric class ID/index per class; in alphabetical order if None
        if class_ids is None:
            class_ids = list(range(len(class_names)))

        self.class_names, self.class_sizes, self.class_ids = map(
            list, zip(*sorted(zip(class_names, class_sizes, class_ids)))
        )

        # Iterate through each class and sample .tif images only; append paths and labels
        self.image_paths = []
        self.labels = []

        for class_id, class_name in zip(self.class_ids, self.class_names):
            # Retrieve all image paths across directories for specified class
            class_paths = []

            for data_subdirectory in self.data_subdirectories:
                class_directory = os.path.join(
                    data_directory, data_subdirectory, class_name
                )

                if os.path.isdir(class_directory):
                    class_paths.extend(
                        [
                            os.path.join(class_directory, filename)
                            for filename in os.listdir(class_directory)
                        ]
                    )

            # Determine new class size and sample images, only include .tif files
            class_idx = class_ids.index(class_id)
            new_class_size = min(self.class_sizes[class_idx], len(class_paths))

            random.seed(self.seed)
            sampled_paths = random.sample(class_paths, new_class_size)

            for image_path in sampled_paths:
                if image_path.lower().endswith(".tif"):
                    try:
                        with Image.open(image_path) as img:
                            img.verify()
                        self.image_paths.append(image_path)
                    except (UnidentifiedImageError, OSError, ValueError):
                        new_class_size -= 1
                else:
                    new_class_size -= 1

            self.class_sizes[class_idx] = new_class_size
            self.labels.extend([class_id] * new_class_size)

        # Other class initializations
        self.image_resolution = image_resolution
        self.image_transforms = image_transforms

    def __len__(self):
        """
        Returns the number of samples in the Dataset.
        """

        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns the image and label of specified sample.

        Args:
            idx (int): Index of specified sample.
        """

        image = Image.open(self.image_paths[idx]).convert("L")
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.image_transforms:
            if label not in self.ignore_classes:
                image = self.aug_transforms(image)
            image = self.image_transforms(image)

        image = image.repeat(3, 1, 1)

        return image, label

    def print_dataset_details(self, indices: list = None, subset_name: str = None):
        """
        Prints the class distribution of the Dataset.

        Args:
            indices (list, optional): Specific indices of subset of Dataset to consider.
            subset_name (str, optional): Specific name of subset of Dataset to print.
        """

        if indices is None:
            filtered_labels = self.labels
        else:
            filtered_labels = [self.labels[i] for i in indices]

        filtered_counts = dict(Counter(filtered_labels))

        if subset_name is None:
            print(f"\nTotal Dataset Size: {len(filtered_labels)}")
        else:
            print(f"\n{subset_name} Dataset Size: {len(filtered_labels)}")

        for class_id, class_name in zip(self.class_indices, self.class_names):
            class_prop = filtered_counts[class_id] / len(filtered_labels)

            print(
                f"Class Name: {class_name} | Class Label: {class_id} | Count: {filtered_counts[class_id]} "
                + f"| Prop: {class_prop:.2f}"
            )

    def print_image_transforms(self):
        """
        Prints the ordered image transformations applied to the Dataset.
        """

        print("\nCurrent Image Transform Pipeline:")
        for tf in self.image_transforms.transforms:
            print(" ", tf)

    def filter_to_class(self, class_id: int):
        """
        Returns the Dataset filtered to a specified class as a Subset object.

        Args:
            class_id (int): ID of class to filter Dataset to.
        """

        filtered_idx = torch.where(torch.tensor(self.labels) == class_id)[0].tolist()

        return Subset(self, filtered_idx)

    def subsample_classes(self, subsample_sizes: dict):
        """
        Samples each class from the Dataset and returns a Subset object.

        Args:
            subsample_sizes (dict): Key-value pairs of class label and specified sample size.
                - Maximum number of items in dict is number of classes in Dataset.
        """

        random.seed(self.seed)

        all_sampled_idx = []

        for class_id, class_sample_size in subsample_sizes.items():
            filtered_idx = torch.where(torch.tensor(self.labels) == class_id)[
                0
            ].tolist()

            if len(filtered_idx) > class_sample_size:
                sampled_idx = random.sample(filtered_idx, class_sample_size)
            else:
                sampled_idx = filtered_idx

            all_sampled_idx.extend(sampled_idx)

        return Subset(self, all_sampled_idx)

    def compute_sample_weights(
        self,
        indices: list = None,
        weights: str = "inverse_weighted",
        normalize_weights: bool = False,
    ):
        """
        Computes weights per class in the Dataset and assigns each sample the corresponding class weight.

        Returns `sample_weights` of length equal to number of samples and
            `class_weights` of length equal to number of classes.

        Args:
            indices (list, optional): Specific indices of subset of Dataset to consider.
            weights (str): Specifies which weights computation to use.
            normalize_weights (bool): Specifies if weights should be normalized.
        """

        labels = torch.tensor(self.labels, dtype=torch.long)
        if indices is not None:
            sub_labels = labels[indices]
        else:
            sub_labels = labels

        class_counts = torch.bincount(
            sub_labels, minlength=len(self.class_names)
        ).float()

        if weights == "balanced":
            class_weights = len(indices) / (class_counts * len(self.class_names))
        elif weights == "inverse_freq":
            class_weights = 1.0 / class_counts
        elif weights == "inverse_square_root":
            class_weights = 1.0 / torch.sqrt(class_counts)
        elif weights == "inverse_log":
            class_weights = 1.0 / torch.log(class_counts + 1.2)  # small constant
        elif weights == "softmax_inverse":
            class_weights = torch.softmax(1.0 / class_counts, dim=0)
        elif weights == "normalized":
            class_weights = class_counts / class_counts.sum()
        else:
            raise ValueError(
                "Unsupported weights computation. Select one of balance, inverse_freq, inverse_square_root, inverse_log, softmax_inverse, or normalized."
            )

        if normalize_weights:
            class_weights = class_weights / class_weights.sum()

        sample_weights = class_weights[sub_labels]

        return sample_weights, class_weights

    def split_train_test_val(
        self,
        train_prop: float = 0.7,
        val_prop: float = 0.1,
        test_prop: float = 0.2,
        verbose: bool = True,
    ):
        """
        Returns indices corresponding to the train, validation and test subsets of the Dataset.

        Args:
            trian_prop (float): Proportion of samples to allocate to the train subset.
            val_prop (float): Proportion of samples to allocate to the validation subset.
            test_prop (float): Proportion of samples to allocate to the test subset.
            verbose (bool): Specifies whether to print distributions of subsets.
        """

        train_split, val_split, test_split = random_split(
            range(len(self)),
            lengths=[train_prop, val_prop, test_prop],
            generator=torch.Generator().manual_seed(self.seed),
        )

        if verbose:
            self.print_dataset_details(train_split.indices, "Train")
            self.print_dataset_details(val_split.indices, "Validation")
            self.print_dataset_details(test_split.indices, "Test")

        return train_split.indices, val_split.indices, test_split.indices

    def append_image_transforms(
        self,
        image_transforms: transforms.Compose = None,
        replace: bool = False,
        verbose: bool = False,
    ):
        """
        Appends image transformations to existing transformation pipeline or replaces.
        If multiple `ToTensor()` transformations are included in the resulting pipeline, only the last instance is kept.
        If there are no `ToTensor()` transformations in the resulting pipeline, it is appended.

        Args:
            image_transforms(transfors.Compose, optional): Iterable of image transformations to append.
            replace (bool): Specifies whether to replace with or append the above image_transforms.
            verbose (bool): Specifies whether to print the resulting image transformation pipeline.
        """

        if image_transforms is None:
            if self.image_transforms is None:
                image_transforms_list = []
            else:
                image_transforms_list = self.image_transforms.transforms
        else:
            if replace:
                image_transforms_list = image_transforms.transforms
            else:
                image_transforms_list = (
                    self.image_transforms.transforms + image_transforms.transforms
                )

        image_transforms_cleaned = []
        to_tensor_indices = [
            i
            for i, tf in enumerate(image_transforms_list)
            if isinstance(tf, transforms.ToTensor)
        ]

        if to_tensor_indices:
            last_idx = to_tensor_indices[-1]
            image_transforms_cleaned = [
                tf
                for i, tf in enumerate(image_transforms_list)
                if not isinstance(tf, transforms.ToTensor) or i == last_idx
            ]
        else:
            image_transforms_cleaned = image_transforms_list + [transforms.ToTensor()]

        self.image_transforms = transforms.Compose(image_transforms_cleaned)

        if verbose:
            self.print_image_transforms()

    def create_dataloaders(
        self,
        batch_size: int,
        train_indices,
        val_indices,
        test_indices,
        image_transforms: transforms.Compose = None,
        transform_val: bool = False,
        train_sample_weights: torch.tensor = None,
    ):
        """
        Creates the train, validatinon and test DataLoaders required for training a PyTorch model.
        If `train_sample_weights` is specified, they are supplied to WeightedRandomSampler for the train subset.

        Args:
            batch_size (int): Sizes of batches to process samples in DataLoader.
            train_indices (list): Indices corresponding to the train subset of the Dataset.
            val_indices (list): Indices corresponding to the validation subset of the Dataset.
            test_indices (list): Indices corresponding to the test subset of the Dataset.
            image_transforms (transforms.Compose, optional): Additional image transformations for the train subset.
            transform_val (bool): Specifies whether to apply train image transformations to the validation subset.
            train_sample_weights (torch.tensor, optional): Contains weights for each sample in the train subset.
        """

        if image_transforms is not None:
            dataset_aug = copy.deepcopy(self)
            dataset_aug.append_image_transforms(
                image_transforms=image_transforms, verbose=False
            )
            train_dataset = Subset(dataset_aug, train_indices)
            if transform_val:
                val_dataset = Subset(dataset_aug, val_indices)
            else:
                val_dataset = Subset(self, val_indices)
        else:
            train_dataset = Subset(self, train_indices)
            val_dataset = Subset(self, val_indices)

        test_dataset = Subset(self, test_indices)

        if train_sample_weights is None:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(self.seed),
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=WeightedRandomSampler(
                    train_sample_weights,
                    num_samples=len(train_sample_weights),
                    replacement=True,
                ),
            )

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, sampler=SequentialSampler(val_dataset)
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, sampler=SequentialSampler(test_dataset)
        )

        return train_loader, val_loader, test_loader
