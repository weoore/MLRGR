import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import streamlit as st

def load_and_predict(model_path, X):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model.predict(X)

def display_prediction_result(prediction):
    if prediction == 0:
        st.success("Транзакция вероятно не является мошеннической")
    else:
        st.success("Транзакция вероятно мошенническая")

df = pd.read_csv('card_tr.csv')

if df is not None:
    st.header("Датасет")
    st.dataframe(df)
    st.write("---")

    df = df.drop('Unnamed: 0', axis=1)

    st.title("Модель классификации мошенничества с кредитными картами")
    list_values = []

    for i in df.columns[:-1]:
        min_value = int(df[i].min())
        max_value = int(df[i].max())

        # Проверка на случай, если min больше или равно max
        if min_value >= max_value:
            st.write("---")
        else:
            # Отображение ползунка только при корректных значениях min и max
            value = st.slider(i, min_value, max_value, max_value // 2)
            list_values.append(value)

    if list_values:
        X_input = np.array(list_values).reshape(1, -1)

        st.title("Тип модели обучения")
        model_type = st.selectbox("Выберите тип", ['Knn', 'Kmeans', 'Boosting', 'Bagging', 'Stacking', 'MLP'])

        button_clicked = st.button("Предсказать")
        if button_clicked and model_type is not None:
            model_path = f'models/{model_type.lower()}.pkl'
            prediction = load_and_predict(model_path, X_input)
            display_prediction_result(prediction)
