import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="ML Interactive Models",
                   layout="centered")

# loading the saved models
DTC_model = pickle.load(open('Decision Tree Classifier.sav', 'rb'))
DTR_model = pickle.load(open('Decision Tree Regressor.sav', 'rb'))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('ML Models',
                           [
                               'Decision Trees',
                               'Linear Regression',
                               'Logistic Regression',
                               'MLP Models',
                               'CNN Models',
                               'KNN Models',
                               'SVM Models',
                            ],
                           default_index=0)


# Decison Tree Page
if selected == 'Decision Trees':

    # page title
    st.title('Decision Tree Classifier')
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
    diab_diagnosis = ''

    # creating a button for Prediction
    if st.button('Predict'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DTreePedigreeFunction, Age]
        user_input = [float(x) for x in user_input]
        diab_prediction = DTC_model.predict([user_input])
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(f"Prediction: {diab_diagnosis}")

    # code block
    st.title('Notebook')
    code = '''
    def hello():
        print("Hello, Streamlit!")
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
    diab_diagnosis = ''

    # creating a button for Prediction
    if st.button('Predict'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DTreePedigreeFunction, Age]
        user_input = [float(x) for x in user_input]
        diab_prediction = DTC_model.predict([user_input])
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(f"Prediction: {diab_diagnosis}")

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
    diab_diagnosis = ''

    # creating a button for Prediction
    if st.button('Predict'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DTreePedigreeFunction, Age]
        user_input = [float(x) for x in user_input]
        diab_prediction = DTC_model.predict([user_input])
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(f"Prediction: {diab_diagnosis}")

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
    diab_diagnosis = ''

    # creating a button for Prediction
    if st.button('Predict'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DTreePedigreeFunction, Age]
        user_input = [float(x) for x in user_input]
        diab_prediction = DTC_model.predict([user_input])
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(f"Prediction: {diab_diagnosis}")

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
    diab_diagnosis = ''

    # creating a button for Prediction
    if st.button('Predict'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DTreePedigreeFunction, Age]
        user_input = [float(x) for x in user_input]
        diab_prediction = DTC_model.predict([user_input])
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(f"Prediction: {diab_diagnosis}")

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
    diab_diagnosis = ''

    # creating a button for Prediction
    if st.button('Predict'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DTreePedigreeFunction, Age]
        user_input = [float(x) for x in user_input]
        diab_prediction = DTC_model.predict([user_input])
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(f"Prediction: {diab_diagnosis}")

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
    diab_diagnosis = ''

    # creating a button for Prediction
    if st.button('Predict'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DTreePedigreeFunction, Age]
        user_input = [float(x) for x in user_input]
        diab_prediction = DTC_model.predict([user_input])
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(f"Prediction: {diab_diagnosis}")

    # code block
    st.title('Notebook')
    code = '''
    def hello():
        print("Hello, Streamlit!")
    '''
    st.code(code, language='python')
