"""
Importing the necessary libraries
"""

import numpy as np
import pandas as pd
import streamlit as st

from scipy import stats
from scipy.special import xlogy

"""
Utility functions for the Model
"""

def entropy(Y: pd.Series, bins: int = 50) -> float:
    """
    Y: pd.Series: Output values

    Returns: float: Entropy
    """

    if Y.empty:
        return 0

    # Y = pd.cut(Y, bins=bins)
    vals = Y.value_counts(normalize=True)
    return -np.sum(xlogy(vals, vals))

def gini_index(Y: pd.Series, bins: int = 50) -> float:
    """
    Y: pd.Series: Output values

    Returns: float: Gini Index
    """

    if Y.empty:
        return 0

    # Y = pd.cut(Y, bins=bins)
    vals = Y.value_counts(normalize=True)
    return 1 - np.sum(np.square(vals))

def information_gain(parent: pd.Series, left: pd.Series, right: pd.Series, criterion: str) -> float:
    """
    parent: pd.Series: Input parent dataset.
    left: pd.Series: Subset of the parent dataset.
    right: pd.Series: Subset of the parent dataset.

    Returns: float: Information gain.
    """
    FMap = {"entropy": entropy, "gini": gini_index}

    # calculate parent and child entropy
    before_entropy = FMap[criterion](parent)
    after_entropy = (len(left) / len(parent)) * FMap[criterion](left) + (len(right) / len(parent)) * FMap[criterion](right)
        
    # calculate information gain 
    information_gain = before_entropy - after_entropy
    return information_gain

def best_split(dataset: pd.DataFrame, num_samples: int, num_features: int, criterion: str) -> dict:
    """
    dataset: pd.DataFrame: The dataset to split.
    num_samples: int: The number of samples in the dataset.
    num_features: int: The number of features in the dataset.

    Returns: dict: A dictionary with the best split.
    """
        
    # Find the best split
    best_split = {'gain': -1, 'feature': None, 'threshold': None, "left_dataset": None, "right_dataset": None}
    for feature_index in range(num_features):
        feature_values = dataset.iloc[:, feature_index]
        thresholds = np.unique(feature_values)
        for threshold in thresholds:
            left_dataset, right_dataset = split_data(dataset, feature_index, threshold)
            y, left_y, right_y = dataset.iloc[:, -1], left_dataset.iloc[:, -1], right_dataset.iloc[:, -1]
            gain = information_gain(y, left_y, right_y, criterion)
            if gain > best_split["gain"]:
                best_split["gain"] = gain
                best_split["feature"] = feature_index
                best_split["threshold"] = threshold
                best_split["left_dataset"] = left_dataset
                best_split["right_dataset"] = right_dataset
    return best_split

def split_data(dataset: pd.DataFrame, feature: int, threshold: float) -> tuple:
    """
    dataset: pd.DataFrame: Input dataset.
    feature: int: Index of the feature to be split on.
    threshold: float: Threshold value to split the feature on.

    Returns:
        left_dataset: pd.DataFrame: Subset of the dataset.
        right_dataset: pd.DataFrame: Subset of the dataset.
    """
    
    # Create mask of the dataset using threshold
    mask = (dataset.iloc[:, feature] <= threshold)

    # Mask the dataset
    left_dataset = dataset[mask]
    right_dataset = dataset[~mask]
    return left_dataset, right_dataset

def bootstrapping(dataset: pd.DataFrame):
    """
    dataset: pd.DataFrame: Input dataset.

    Returns:
        dataset: pd.DataFrame: Bootstrapped dataset.
    """

    # np.random.seed(42)
    n_samples = dataset.shape[0]

    indices = np.random.choice(n_samples, n_samples, replace=True)
    return dataset.iloc[indices]

class Node():
    """
    A class representing a node in a decision tree.
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None, gain=None, value=None):
        """
        feature: string: The feature used for splitting at this node.
        threshold: List of float: The threshold used for splitting at this node.
        left: Node: Pointer to the left Node.
        Right: Node: Pointer to the Right Node.
        gain: float: The gain of the split.
        value: float: predicted value at this node.
        """

        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value

class DecisionTree():
    """
    A decision tree classifier/regressor.
    """

    def __init__(self, type: str, criterion: str = 'entropy', min_samples: int = 2, max_depth: int = 2):
        """
        Constructor for DecisionTree class.

        type: string: The type of the decision tree.
        criterion: string: The criterion used to split nodes.
        min_samples: int: Minimum number of samples at leaf node.
        max_depth: int: Maximum depth of the decision tree.
        """

        if type not in ["classification", "regression"]:
            raise ValueError("type should be either 'classification' or 'regression'")
        
        if criterion not in ["entropy", "gini"]:
            raise ValueError("criterion should be either 'entropy' or 'gini'")
        
        self.type = type
        self.criterion = criterion
        self.min_samples = min_samples
        self.max_depth = max_depth
    
    def build_tree(self, dataset: pd.DataFrame, current_depth: int = 0) -> Node:
        """
        dataset: pd.DataFrame: The dataset to build the tree.
        current_depth: int: The current depth of the tree.

        Returns: Node: The root node of the decision tree.
        """
        
        # split the dataset into X, y values
        X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
        n_samples, n_features = X.shape
        
        # Terminating conditions
        if n_samples >= self.min_samples and current_depth <= self.max_depth:
            best_split_values = best_split(dataset, n_samples, n_features, self.criterion)
            left_node = self.build_tree(best_split_values["left_dataset"], current_depth + 1)
            right_node = self.build_tree(best_split_values["right_dataset"], current_depth + 1)

            return Node(best_split_values["feature"], best_split_values["threshold"], left_node, right_node, best_split_values["gain"])

        # compute leaf node value
        if self.type == "classification":
            leaf_value = -1 if y.empty else y.mode()[0]
        elif self.type == "regression":
            leaf_value = -1 if y.empty else y.median()
        
        return Node(value=leaf_value)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        X: pd.DataFrame: The feature datset.
        y: pd.Series: The target values.
        """
        
        dataset = pd.concat([X, y], axis=1) 
        self.root = self.build_tree(dataset)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        X: pd.DataFrame: The feature matrix to make predictions for.

        Returns:
            predictions: pd.Series: A Series of predicted class labels.
        """
        
        predictions = X.apply(self.traverse_tree, axis=1, args=(self.root,))
        return predictions
    
    def traverse_tree(self, X: pd.Series, node: Node) -> float:
        """
        X: pd.Series: The feature vector to predict the target value for.
        node: Node: The current node being evaluated.

        Returns: float: The predicted target value.
        """
        
        if node.value is not None: # if the node is a leaf node
            return node.value
        else: # if the node is not a leaf node
            feature = X.iloc[node.feature]
            if feature <= node.threshold:
                return self.traverse_tree(X, node.left)
            else:
                return self.traverse_tree(X, node.right)
            
    def plot_tree(self, node: Node = None, depth: int = 0):
        """
        Plot the decision tree.

        node: Node: The current node being evaluated.
        depth: int: The current depth of the tree.
        """
        
        if node is None:
            node = self.root
        
        if node.value is not None:
            st.write(f"{'&nbsp;&nbsp;&nbsp;&nbsp;' * (4 * depth)}Predict: {round(node.value, 3)}")
        else:
            st.write(f"{'&nbsp;&nbsp;&nbsp;&nbsp;' * (4 * depth)}?(column {node.feature} <= {round(node.threshold, 3)})")
            self.plot_tree(node.left, depth + 1)
            self.plot_tree(node.right, depth + 1)

class RandomForest():
    """
    A Random Forest classifier/regressor.
    """

    def __init__(self, type: str, criterion: str = 'entropy', n_trees: int = 2, min_samples: int = 2, max_depth: int = 2):
        """
        Constructor for Random Forest class.

        type: string: The type of the decision tree.
        criterion: string: The criterion used to split nodes.
        n_trees: int: Number of trees in random forest.
        min_samples: int: Minimum number of samples at leaf node.
        max_depth: int: Maximum depth of the decision tree.
        """

        if type not in ["classification", "regression"]:
            raise ValueError("type should be either 'classification' or 'regression'")
        
        if criterion not in ["entropy", "gini"]:
            raise ValueError("criterion should be either 'entropy' or 'gini'")
        
        self.type = type
        self.criterion = criterion
        self.n_trees = n_trees
        self.trees = []
        self.min_samples = min_samples
        self.max_depth = max_depth

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        X: pd.DataFrame: The feature datset.
        y: pd.Series: The target values.
        """
        
        dataset = pd.concat([X, y], axis=1)

        for _ in range(self.n_trees):
            tree = DecisionTree(type=self.type, criterion=self.criterion, min_samples=self.min_samples, max_depth=self.max_depth)

            dataset_bootstrap = bootstrapping(dataset)
            X, y = dataset_bootstrap.iloc[:, :-1], dataset_bootstrap.iloc[:, -1]

            tree.fit(X, y)
            self.trees.append(tree)

    def predict(self, X: pd.DataFrame):
        """
        X: pd.DataFrame: The feature matrix to make predictions for.

        Returns:
            predictions: pd.Series: A Series of predicted class labels.
        """
        
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        if self.type == 'classification':
            predictions = stats.mode(predictions, axis=0)[0]
        else:
            predictions = np.mean(predictions, axis=0)

        return pd.Series(predictions.reshape(-1))
    
    def plot_forest(self, n_plot: int = None):
        """
        Plot the random forest.
        n_plot: int: The number of trees to plot.
        """

        if n_plot == None:
            n_plot = len(self.trees)
        else:
            n_plot = min(n_plot, len(self.trees))

        for i in range(n_plot):
            print(f"Decision Tree {i}\n")
            self.trees[i].plot_tree()
            print("\n------------------------------------------------------------\n")