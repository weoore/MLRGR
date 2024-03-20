import io
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import NearMiss

def load_and_predict(model_path, X):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model.predict(X)

def display_prediction_result(prediction):
    if prediction == 0:
        st.success("Транзакция вероятно не является мошеннической")
    else:
        st.success("Транзакция вероятно мошенническая")

def main():
    data = st.file_uploader("Выберите файл датасета", type=["csv"])

    if data is not None:
        st.header("Датасет")
        df = pd.read_csv(data)
        st.dataframe(df)
        st.write("---")

        feature = st.selectbox("Выберите предсказываемый признак", df.columns)

        st.title("Тип модели обучения")
        model_type = st.selectbox("Выберите тип", ['Knn', 'decisiontree', 'Boosting', 'Bagging', 'Stacking', 'gradientboosting', 'MLP'])

        button_clicked = st.button("Обработка данных и предсказание")
        if button_clicked:
            st.header("Обработка данных")

            df = df.drop_duplicates()
            df.fillna(df.mean(), inplace=True)  # Заполнение NaN средними значениями по столбцам

            scaler = StandardScaler()
            data_scaler = scaler.fit_transform(df.drop(feature, axis=1))

            y = df[feature]
            X = df.drop([feature], axis=1)
            X, _, y, _ = train_test_split(X, y, test_size=0.9, random_state=42)

            nm = NearMiss()
            X, y = nm.fit_resample(X, y.ravel())

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

            st.success("Обработка завершена")

            st.header("Предсказание")

            if model_type != "Выберите модель":
                model_path = f'models/{model_type.lower()}.pkl'
                prediction = load_and_predict(model_path, X_test)
                display_prediction_result(prediction)

if __name__ == "__main__":
    main()
