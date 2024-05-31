import os
import pickle
import torch
from torchtext.data.utils import get_tokenizer
from torchvision.transforms import ToTensor, ToPILImage

import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_image_comparison import image_comparison

import numpy as np
import pandas as pd
import string
from PIL import Image
from sklearn import preprocessing
from DT_RF_Models import Node, DecisionTree, RandomForest
from LR_LR_Models import LinearRegression
from LSTM_Models import MLP
from KNN_Models import KMeans

# Set page configuration
st.set_page_config(page_title="ML Interactive Models",
                   layout="centered")

# loading the saved models
Models_dir = "Models"
Models = {
    'Decision Trees': {},
    'Random Forests': {},
    'Linear Regression': {},
    'Logistic Regression': {},
    'LSTM Models': {},
    'CNN Models': {},
    'KNN Models': {},
    'SVM Models': {},
}

for model_name in os.listdir(Models_dir):
    with open(os.path.join(Models_dir, model_name), 'rb') as file:
        if model_name.startswith('DTree'):
            model = pickle.load(file)
            Models['Decision Trees'][model_name] = model
        elif model_name.startswith('RForest'):
            model = pickle.load(file)
            Models['Random Forests'][model_name] = model
        elif model_name.startswith('Linear'):
            model = LinearRegression.load_model(file.name) if model_name.endswith('.pth') else pickle.load(file)
            Models['Linear Regression'][model_name] = model
        elif model_name.startswith('Logistic'):
            Models['Logistic Regression'][model_name] = model
        elif model_name.startswith('LSTM'):
            model = MLP.load_model(file.name) if model_name.endswith('.pth') else pickle.load(file)
            Models['LSTM Models'][model_name] = model
        elif model_name.startswith('CNN'):
            Models['CNN Models'][model_name] = model
        elif model_name.startswith('KNN'):
            Models['KNN Models'][model_name] = model
        elif model_name.startswith('SVM'):
            Models['SVM Models'][model_name] = model

# sidebar for navigation
with st.sidebar:
    selected = option_menu('ML Models',
                           [
                               'MLP Models',
                               'KNN Models',
                               'Decision Tree',
                               'Random Forest',
                               'Linear Regression',
                            #    'Logistic Regression',
                            #    'CNN Model',
                            #    'SVM Model',
                            ],
                           default_index=0)


# Decison Tree Page
if selected == 'Decision Tree':

    # page title
    st.title('Decision Tree Classifier')
    st.write('This is a Decison Tree model for Breast Cancer')

    # dataset link
    st.markdown(
        """
        <a href="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic" target="_blank">
            <button style='background-color: #262730;
            border: 0px;
            border-radius: 10px;
            color: white;
            padding: 10px 15px;
            text-align: center;
            text-decoration: none;
            font-size: 16px;
            margin-bottom: 1rem;
            cursor: pointer;'>Breast Cancer Dataset</button>
        </a>
        """, 
        unsafe_allow_html=True,
    )

    # creating input fields for depth and criteria
    depth = st.slider('Depth', min_value=0, max_value=4, value=1)
    criteria = st.selectbox('Criteria', options=['Entropy', 'Gini'])

    # getting the input data from the user
    user_input =  ['radius1', 26.57, 142.70, 'area1', 'smoothness1', 'compactness1', 0.24870, 0.04960, 'radius2', 2.608, 'area2', 0.015600, 10.01, 19.23, 65.59, 310.1, 'smoothness3', 'compactness3', 0.3791, 0.15140, 0.2837]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        user_input[0] = st.text_input('Radius 1 (cm)')
    with col2:
        user_input[3] = st.text_input('Area 1 (sqcm)')
    with col3:
        user_input[4] = st.text_input('Smoothness 1 (0-1)')
    with col4:
        user_input[5] = st.text_input('Compactness 1 (0-1)')
    with col1:
        user_input[8] = st.text_input('Radius 2 (cm)')
    with col2:
        user_input[10] = st.text_input('Area 2 (sqcm)')
    with col3:
        user_input[16] = st.text_input('Smoothness 2 (0-1)')
    with col4:
        user_input[17] = st.text_input('Compactness 2 (0-1)')

    # code for Prediction
    diagnosis = ''

    # creating a button for Prediction
    if st.button('Predict'):
        user_input = pd.DataFrame([float(x) for x in user_input]).T
        prediction = Models['Decision Trees'][f'DTree_{criteria}_{depth}.pkl'].predict(user_input)
        if prediction[0] == 1:
            diagnosis = 'Malignant'
        else:
            diagnosis = 'Benign'

    st.success(f"Diagnosis: {diagnosis}")

    # plot the decision tree
    # st.write(f"Decision Tree with criteria {criteria} and depth {depth}")
    # Models['Decision Trees'][f'DTree_{criteria}_{depth}.pkl'].plot_tree()

    # code block
    st.title('Notebook')
    code = '''
    """
    Utility functions for the Model
    """

    def entropy(Y: pd.Series) -> float:
        """
        Y: pd.Series: Output values

        Returns: float: Entropy
        """

        vals = Y.value_counts(normalize=True)
        return -np.sum(xlogy(vals, vals))

    def gini_index(Y: pd.Series) -> float:
        """
        Y: pd.Series: Output values

        Returns: float: Gini Index
        """

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
    '''
    st.code(code, language='python')
    code = '''
    class Node():
        """
        A class representing a node in a decision tree.
        """

        def __init__(self, feature=None, threshold=None, left=None, right=None, gain=None, value=None):
            """
            feature: string: The feature used for splitting at this node.
            threshold: float: The threshold used for splitting at this node.
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
    '''
    st.code(code, language='python')
    code = '''
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
        
        def build_tree(self, dataset: pd.DataFrame, current_depth: int = 0):
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

        def predict(self, X: pd.DataFrame):
            """
            X: pd.DataFrame: The feature matrix to make predictions for.

            Returns:
                predictions: pd.Series: A Series of predicted class labels.
            """
            
            predictions = X.apply(self.traverse_tree, axis=1, args=(self.root,))
            return predictions
        
        def traverse_tree(self, X: pd.Series, node: Node):
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
                print(f"{4*depth * '  '}Predict: {round(node.value, 3)}")
            else:
                print(f"{4*depth * '  '}?(column {node.feature} <= {round(node.threshold, 3)})")
                self.plot_tree(node.left, depth + 1)
                self.plot_tree(node.right, depth + 1)
        '''
    st.code(code, language='python')


# Random Forest Page
if selected == 'Random Forest':

    # page title
    st.title('Random Forest Classifier')
    st.write('This is a Random Forest model for Algerian Forest Fires')

    # dataset link
    st.markdown(
        """
        <a href="https://archive.ics.uci.edu/dataset/547/algerian+forest+fires+dataset" target="_blank">
            <button style='background-color: #262730;
            border: 0px;
            border-radius: 10px;
            color: white;
            padding: 10px 15px;
            text-align: center;
            text-decoration: none;
            font-size: 16px;
            margin-bottom: 1rem;
            cursor: pointer;'>Algerian Forest Fires Dataset</button>
        </a>
        """, 
        unsafe_allow_html=True,
    )

    # creating input fields for depth, n_trees and criteria
    depth = st.slider('Depth', min_value=0, max_value=5, value=1)
    n_trees = st.slider('N Trees', min_value=1, max_value=5, value=3)
    criteria = st.selectbox('Criteria', options=['Entropy', 'Gini'])


    # getting the input data from the user
    user_input =  ['Temperature', ' RH', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']

    col1, col2, col3 = st.columns(3)
    with col1:
        user_input[0] = st.text_input('Temperature (C)')
    with col2:
        user_input[1] = st.text_input('Relative Humidity (%)')
    with col3:
        user_input[2] = st.text_input('Rain (mm)')
    with col1:
        user_input[3] = st.text_input('Fine Fuel Moisture Code (index)')
    with col2:
        user_input[4] = st.text_input('Duff Moisture Code (index)')
    with col3:
        user_input[5] = st.text_input('Drought Code (index)')
    with col1:
        user_input[6] = st.text_input('Initial Spread Index (index)')
    with col2:
        user_input[7] = st.text_input('Buildup Index (index)')
    with col3:
        user_input[8] = st.text_input('Fire Weather Index (index)')

    # code for Prediction
    predict = ''

    # creating a button for Prediction
    if st.button('Predict'):
        user_input = pd.DataFrame([float(x) for x in user_input]).T
        prediction = Models['Random Forests'][f'RForest_{criteria}_{n_trees}_{depth}.pkl'].predict(user_input)
        if prediction[0] == 1:
            predict = 'Fire'
        else:
            predict = 'Not Fire'

    st.success(f"Prediction: {predict}")

    # code block
    st.title('Notebook')
    code = '''
    """
    Utility functions for the Model
    """

    def entropy(Y: pd.Series) -> float:
        """
        Y: pd.Series: Output values

        Returns: float: Entropy
        """

        vals = Y.value_counts(normalize=True)
        return -np.sum(xlogy(vals, vals))

    def gini_index(Y: pd.Series) -> float:
        """
        Y: pd.Series: Output values

        Returns: float: Gini Index
        """

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
    '''
    st.code(code, language='python')
    code = '''
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
                print(f"Decision Tree {i}\\n")
                self.trees[i].plot_tree()
                print("\\n------------------------------------------------------------\\n")
    '''
    st.code(code, language='python')


# Linear Regression Page
if selected == 'Linear Regression':

    # page title
    st.title('Linear Regression')
    st.write('This is a Linear Regression model for MPG Prediction')

    # dataset link
    st.markdown(
        """
        <a href="https://archive.ics.uci.edu/dataset/9/auto+mpg" target="_blank">
            <button style='background-color: #262730;
            border: 0px;
            border-radius: 10px;
            color: white;
            padding: 10px 15px;
            text-align: center;
            text-decoration: none;
            font-size: 16px;
            margin-bottom: 1rem;
            cursor: pointer;'>Auto MPG Dataset</button>
        </a>
        """, 
        unsafe_allow_html=True,
    )

    # creating input fields for learning rate and epochs
    lr = st.select_slider('Learning rate', options=[0.001, 0.01, 0.1], value=0.01)
    epochs = st.select_slider('Epochs', options=list(range(10, 101, 10)), value=70)

    # getting the input data from the user
    user_input =  ['displacement', 'cylinders', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']

    col1, col2, col3 = st.columns(3)
    with col1:
        user_input[0] = st.text_input('Displacement (m)', '150')
    with col2:
        user_input[1] = st.text_input('Cylinders (unit)', '6')
    with col3:
        user_input[2] = st.text_input('Horsepower (Watts)', '100')
    with col1:
        user_input[3] = 2.204*float(st.text_input('Weight (Kg)', '1300'))
    with col2:
        user_input[4] = st.text_input('Acceleration (0-100Km/h in sec)', '15')
    with col3:
        user_input[5] = st.text_input('Model Year (1970 - 1982)', '76')
    with col1:
        origin = {'USA': 1, 'Europe': 2, 'Asia': 3}
        user_input[6] = origin[st.text_input('Origin (USA, Europe, Asia)', 'Asia')]

    # code for Prediction
    predict = ''

    # creating a button for Prediction
    if st.button('Predict'):
        user_input = pd.DataFrame([float(x) for x in user_input]).T
        user_input = Models['Linear Regression']['Linear_StandardScaler.pkl'].transform(user_input)
        user_input = torch.tensor(user_input, dtype=torch.float32)
        predict = Models['Linear Regression'][f'Linear_{lr}_{epochs}.pth'].predict(user_input).squeeze()
        predict = "{:.1f}".format(predict)

    st.success(f"Prediction: {predict}")

    # code block
    st.title('Notebook')
    code = '''
    class LinearRegression(nn.Module):
        """
        A Linear Regressor.
        """

        def __init__(self, in_features: int, out_features: int, learning_rate: float = 0.01, random_state: int = None):
            """
            Constructor for Linear Regression.

            in_features: int: The number of input features.
            out_features: int: The number of output features.
            random_state: int: The seed for the random number generator.
            """
            
            super(LinearRegression, self).__init__()
            if random_state is not None:
                np.random.seed(random_state)
            self.W = torch.tensor(np.random.randn(out_features, in_features), dtype=torch.float32).to(device)
            self.b = torch.tensor(np.random.randn(1, out_features), dtype=torch.float32).to(device)
            self.lr = learning_rate
            self.in_features = in_features
            self.out_features = out_features

        def forward(self, X: torch.Tensor) -> torch.Tensor:
            """
            X: torch.Tensor: The input tensor.
            """

            return torch.matmul(X, self.W.T) + self.b
        
        def fit(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 100, print_cost: bool = False):
            """
            X: torch.Tensor: The input tensor.
            """

            X, y = X.reshape(-1, self.in_features).to(device), y.reshape(-1, self.out_features).to(device)

            Cost = []
            for i in range(epochs):
                # Forward pass
                predictions = self.forward(X)
                cost = torch.mean((predictions - y) ** 2)/2
                Cost.append(cost.item())

                # Backward pass
                dW = torch.matmul((predictions - y).T, X)/X.shape[0]
                db = torch.mean(predictions - y, dim=0, keepdim=True)

                # Update the weights
                self.W = (self.W - self.lr * dW).to(device)
                self.b = (self.b - self.lr * db).to(device)

                # Print the cost
                if print_cost and (i+1) % 100 == 0:
                    print(f'Cost at epoch {i+1}: {round(cost.item(), 3)}')
                    print("\\n------------------------------------------------------------\\n")

            return Cost
        
        def predict(self, X: torch.Tensor) -> torch.Tensor:
            """
            X: torch.Tensor: The input tensor.
            """

            return self.forward(X)

        def save_model(self, path):
            """
            Save the model parameters.

            path: str: The path where the model parameters should be saved.
            """
            torch.save({
                'W': self.W,
                'b': self.b,
                'lr': self.lr,
                'in_features': self.in_features,
                'out_features': self.out_features
            }, path)

        @staticmethod
        def load_model(path):
            """
            Load the model parameters.

            path: str: The path from where the model parameters should be loaded.
            """
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            model = LinearRegression(in_features=state_dict['in_features'], out_features=state_dict['out_features'], learning_rate=state_dict['lr'])
            model.W = state_dict['W']
            model.b = state_dict['b']
            return model
    '''
    st.code(code, language='python')


# Logistic Regression Page
if selected == 'Logistic Regression':

    # page title
    st.title('Logistic Regression')
    st.write('This is a Decison Tree model for Diabetes Prediction')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DTreePedigreeFunction = st.text_input('DTree Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')


    # code for Prediction
    diagnosis = ''

    # creating a button for Prediction
    if st.button('Predict'):
        user_input = pd.DataFrame([float(x) for x in user_input]).T
        prediction = Models['Decision Trees'][f'DTree_{criteria}_{depth}.pkl'].predict(user_input)
        if prediction[0] == 1:
            diagnosis = 'Malignant'
        else:
            diagnosis = 'Benign'

    st.success(f"Prediction: {diagnosis}")

    # code block
    st.title('Notebook')
    code = '''
    def hello():
        print("Hello, Streamlit!")
    '''
    st.code(code, language='python')


# MLP Models Page
if selected == 'MLP Models':

    # page title
    st.title('Multi Layer Perceptron Models')
    st.write('This is a variation of MLPs for Next Word Prediction')

    # creating input fields for learning rate and epochs
    block_size = st.select_slider('Block Size', options=[2, 4, 8, 16, 32], value=4)
    emb_dim = st.select_slider('Embedding Dimensions', options=[2, 4, 8, 16, 32], value=16)
    context = st.select_slider('Context Length', options=range(0, 1001), value=100)
    corpusOptions = ['Start-ups', 'Wikipedia']
    corpus = st.selectbox('Select a corpus', corpusOptions)

    # getting encodings and decodings
    model = Models['LSTM Models'][f'LSTM_{corpus}_{block_size}_{emb_dim}.pth']
    encodings = Models['LSTM Models'][f'LSTM_{corpus}_encodings.pkl']
    decodings = Models['LSTM Models'][f'LSTM_{corpus}_decodings.pkl']

    # getting the input data from the user
    input_text = st.text_area('Enter your Text:')

    # Initialize the tokenizer
    tokenizer = get_tokenizer('basic_english')
    input_tokens = [tkn for tkn in tokenizer(input_text) if tkn]


    # creating a button for Prediction
    if st.button('Predict'):
        output_text = input_text

        # Process the input
        if len(input_tokens) > model.block_size:
            input_tokens = input_tokens[-model.block_size:]
        elif len(input_tokens) < model.block_size:
            input_tokens = ['<unk>'] * (model.block_size - len(input_tokens)) + input_tokens
        input_encoded = torch.tensor([encodings.get(token, encodings['<unk>']) for token in input_tokens])

        # Create a placeholder for the output text
        st.write('Final Generated Text:')
        output_placeholder = st.empty()

        for token in model.predict(input_encoded, decodings, context):
            if token != '<unk>':
                if token in string.punctuation:
                    output_text += token
                else:
                    output_text += ' ' + token
            
            output_placeholder.write(output_text)

    # code block
    st.title('Notebook')
    code = '''
    class LSTM(nn.Module):
        """
        A Long Short Term Memory.
        """

        def __init__(self, block_size: int, vocab_size: int, emb_dim: int, random_state: int = None):
            """
            Constructor for Long Short Term Memory.

            block_size: int: input block size
            vocab_size: int: vocabulary of the embedded words
            emd_dim: int: embedding dimension of the characters
            random_state: int: random state for reproducibility
            """

            super(MLP, self).__init__()
            if random_state is not None:
                torch.manual_seed(random_state)
            self.block_size = block_size
            self.vocab_size = vocab_size
            self.emb_dim = emb_dim
            self.embeddings = nn.Sequential(
                nn.Embedding(vocab_size, emb_dim),
                nn.Flatten()
            )
            self.layers = nn.Sequential(
                nn.Linear(block_size * emb_dim, 256),
                nn.SiLU(),
                nn.Linear(256, 32),
                nn.SiLU(),
                nn.Linear(32, vocab_size)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            x: torch.Tensor: The input tensor.
            """

            x = self.embeddings(x)
            x = self.layers(x)
            return x

        def fit(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 1000, batch_size: int = 4096, learning_rate: float = 0.01, print_cost: bool = False):
            """
            X: torch.Tensor: The input tensor
            y: torch.Tensor: The target tensor
            epochs: int: The number of epochs
            batch_size: int: The batch size while applying mini-batch gradient descent
            learning_rate: float: learning rate of the optimizer
            print_cost: bool: Whether to print the cost or not
            """
            self.lr = learning_rate

            X, y = X.reshape(-1, self.block_size).to(device), y.reshape(-1).to(device)
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

            Losses = []
            for i in range(epochs):
                for batch_X, batch_y in dataloader:
                    # Forward pass
                    predictions = self.forward(batch_X)
                    loss = criterion(predictions, batch_y)
                    Losses.append(loss.item())

                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                # Print the cost
                if print_cost and (i+1) % 10 == 0:
                    print(f'Loss at epoch {i+1}: {loss.item():.3f}')
                    print("\\n------------------------------------------------------------\\n")

            return Losses

        def predict(self, X: torch.Tensor, decodings: dict, context_len: int):
            """
            X: torch.Tensor: The input tensor
            decodings: dict: The dictionary containing decoding of the characters
            context_len: int: The length of the context
            """

            X = X.reshape(1, self.block_size).to(device)

            for _ in range(context_len):
                y_pred = self.forward(X)
                id_pred = torch.distributions.Categorical(logits=y_pred).sample().item()
                decode = decodings[id_pred]
                X = torch.cat((X[:, 1:], torch.tensor([[id_pred]], device=device)), 1)
                yield decode

        def save_model(self, path):
            """
            Save the model parameters.

            path: str: The path where the model parameters should be saved.
            """

            model_info = {
                'block_size': self.block_size,
                'vocab_size': self.vocab_size,
                'emb_dim': self.emb_dim,
                'state_dict': self.state_dict()
            }

            torch.save(model_info, path)

        @staticmethod
        def load_model(path):
            """
            Load the model parameters.

            path: str: The path from where the model parameters should be loaded.
            """

            model_info = torch.load(path, map_location=torch.device('cpu'))
            model = MLP(block_size=model_info['block_size'], vocab_size=model_info['vocab_size'], emb_dim=model_info['emb_dim'])
            model.load_state_dict(model_info['state_dict'])
            return model
    '''
    st.code(code, language='python')


# CNN Models Page
if selected == 'CNN Models':

    # page title
    st.title('Convolutional Neural Network Models')
    st.write('This is a Decison Tree model for Diabetes Prediction')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DTreePedigreeFunction = st.text_input('DTree Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')


    # code for Prediction
    diagnosis = ''

    # creating a button for Prediction
    if st.button('Predict'):
        user_input = pd.DataFrame([float(x) for x in user_input]).T
        prediction = Models['Decision Trees'][f'DTree_{criteria}_{depth}.pkl'].predict(user_input)
        if prediction[0] == 1:
            diagnosis = 'Malignant'
        else:
            diagnosis = 'Benign'

    st.success(f"Prediction: {diagnosis}")

    # code block
    st.title('Notebook')
    code = '''
    def hello():
        print("Hello, Streamlit!")
    '''
    st.code(code, language='python')


# KNN Models Page
if selected == 'KNN Models':

    # page title
    st.title('K Means Models')
    st.write('Image Clustering using KMeans Algorithm')

    # getting the input data from the user
    clusters = st.slider('Number of Clusters', min_value=2, max_value=32, value=5)
    image = st.file_uploader('Upload an Image:', type=['jpg', 'jpeg'])

    # initializing the model
    model = KMeans(n_clusters = clusters)
    
    if st.button('Cluster the Image') and image is not None:
        image = Image.open(image).convert('RGB')
        T_image = ToTensor()(image)
        model.fit(T_image.reshape(-1, 3).float())

        # get the labels and the segmented image
        labels = model.predict(T_image.reshape(-1, 3).float())
        segmented_image = ToPILImage()(torch.stack([model.centroids[i] for i in labels]).view(T_image.shape))

        # display the images
        image_comparison(
            img1=image,
            img2=segmented_image,
            label1='Original Image',
            label2='Clustered Image',
        )

    # code block
    st.title('Notebook')
    code = '''
    class KMeans:
        """
        A KMeans clustering algorithm.
        """
        def __init__(self, n_clusters: int = 3):
            """
            Constructor for KMeans.

            n_clusters: int: The number of clusters.
            """
            self.n_clusters = n_clusters
            self.centroids = []

        def fit(self, X : torch.Tensor, max_iter : int = 100, tol : float = 1e-4):
            """
            X: torch.Tensor: The data to cluster.
            max_iter: int: The maximum number of iterations.
            tol: float: The tolerance for convergence.
            """

            X = X.to(device)
            self.findCenters(X)
            
            for _ in range(max_iter):
                distances = torch.cdist(X, torch.stack(self.centroids))
                labels = torch.argmin(distances, dim=1)

                centroids = []
                for i in range(self.n_clusters):
                    if (labels == i).any():
                        centroids.append(X[labels == i].mean(dim=0))
                    else:
                        centroids.append(X[np.random.randint(0, X.shape[0])])

                if torch.allclose(torch.stack(centroids), torch.stack(self.centroids), atol=tol):
                    break
                
                self.centroids = centroids

        def predict(self, X):
            """
            X: torch.Tensor: The data to cluster.

            Returns: torch.Tensor: The cluster labels.
            """

            X = X.to(device)
            distances = torch.cdist(X, torch.stack(self.centroids))
            return torch.argmin(distances, dim=1)

        def findCenters(self, X):
            """
            X: torch.Tensor: The data to cluster.
            """

            self.centroids = [X[np.random.randint(0, X.shape[0])]]
            
            for _ in range(self.n_clusters - 1):
                distances = torch.sum(torch.stack([torch.norm(X - c, dim=1) for c in self.centroids]), dim=0)
                self.centroids.append(X[torch.argmax(distances)])
    '''
    st.code(code, language='python')


# SVM Models Page
if selected == 'SVM Models':

    # page title
    st.title('Support Vector Machine Models')
    st.write('This is a Decison Tree model for Diabetes Prediction')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DTreePedigreeFunction = st.text_input('DTree Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')


    # code for Prediction
    diagnosis = ''

    # creating a button for Prediction
    if st.button('Predict'):
        user_input = pd.DataFrame([float(x) for x in user_input]).T
        prediction = Models['Decision Trees'][f'DTree_{criteria}_{depth}.pkl'].predict(user_input)
        if prediction[0] == 1:
            diagnosis = 'Malignant'
        else:
            diagnosis = 'Benign'

    st.success(f"Prediction: {diagnosis}")

    # code block
    st.title('Notebook')
    code = '''
    def hello():
        print("Hello, Streamlit!")
    '''
    st.code(code, language='python')
