import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

import pandas as pd
from DT_RF_Models import Node, DecisionTree, RandomForest

# Set page configuration
st.set_page_config(page_title="ML Interactive Models",
                   layout="centered")

# loading the saved models
Models_dir = "Models"
Models = {
    'Decision Trees': {},
    'Random Forests': {},
}

for model_name in os.listdir(Models_dir):
    with open(os.path.join(Models_dir, model_name), 'rb') as file:
        model = pickle.load(file)
        if model_name.startswith('DTree'):
            Models['Decision Trees'][model_name] = model
        elif model_name.startswith('RForest'):
            Models['Random Forests'][model_name] = model

# sidebar for navigation
with st.sidebar:
    selected = option_menu('ML Models',
                           [
                               'Decision Tree',
                               'Random Forest',
                               'Linear Regression',
                               'Logistic Regression',
                               'MLP Model',
                               'CNN Model',
                               'KNN Model',
                               'SVM Model',
                            ],
                           default_index=0)


# Decison Tree Page
if selected == 'Decision Tree':

    # page title
    st.title('Decision Tree Classifier')
    st.write('This is a Decison Tree model for Breast Cancer')

    # creating input fields for depth and criteria
    depth = st.slider('Depth', min_value=0, max_value=4, value=3)
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

    # code block
    st.title('Notebook')
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
    st.write('This is a Random Forest model for Breast Cancer')

    # creating input fields for depth and criteria
    depth = st.slider('Depth', min_value=0, max_value=4, value=3)
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
        prediction = Models['Random Forests'][f'DTree_{criteria}_{depth}.pkl'].predict(user_input)
        if prediction[0] == 1:
            diagnosis = 'Malignant'
        else:
            diagnosis = 'Benign'

    st.success(f"Diagnosis: {diagnosis}")

    # code block
    st.title('Notebook')
    code = '''
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

# Linear Regression Page
if selected == 'Linear Regression':

    # page title
    st.title('Linear Regression')
    st.write('This is a Linear Regression model for Diabetes Prediction')

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
    st.title('K Nearest Neighbors Models')
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
