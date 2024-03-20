import io
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import NearMiss
import tensorflow as tf

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

    st.title("Ввести наблюдение")

    clear_data = pd.DataFrame({'distance_from_home': [0],
                               'distance_from_last_transaction': [0],
                               'ratio_to_median_purchase_price': [0],
                               'repeat_retailer': [0],
                               'used_chip': [0],
                               'used_pin_number': [0],
                               'online_order': [0],
                               'fraud': [0]})

    uploaded_file = st.file_uploader("Выберите файл набора данных")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Датасет загружен", df)
    else:
        df = clear_data

    input_method = st.selectbox('Выберите способ ввода данных', ['Ручной ввод', 'Случайное наблюдение из датасета'])
    if input_method == 'Ручной ввод':
        st.header("distance_from_home")

        distance_from_home = st.number_input("Расстояние от дома:", value=32)

        st.header("distance_from_last_transaction")
        distance_from_last_transaction = st.number_input("Расстояние от последней транзакции:", value=50)

        st.header("ratio_to_median_purchase_price")
        ratio_to_median_purchase_price = st.number_input("Отношение цены покупки к медианной цене:", value=1.5)

        st.header("repeat_retailer")
        repeat_retailer = st.number_input("Покупка совершенна у одного и того же продавца:", value=1)

        st.header("used_chip")
        used_chip = st.number_input("Использован чип:", value=1)

        st.header("used_pin_number")
        used_pin_number = st.number_input("Введён пин-код:", value=1)

        st.header("online_order")
        online_order = st.number_input("Онлайн заказ:", value=0)

        data = pd.DataFrame({'distance_from_home': [distance_from_home],
                             'distance_from_last_transaction': [distance_from_last_transaction],
                             'ratio_to_median_purchase_price': [ratio_to_median_purchase_price],
                             'repeat_retailer': [repeat_retailer],
                             'used_chip': [used_chip],
                             'used_pin_number': [used_pin_number],
                             'online_order': [online_order],
                             'fraud': None})
    else:
        if df.equals(clear_data):
            st.write("**Датасет не выбран.**")
        data = df.sample(n=1)

    st.title("Сделать новое предсказание")
    button = st.button("Предсказать, является ли транзакция мошеннической")
    if button:
        st.write(data)
        data = data.drop(columns='fraud')
        with open('./models/knn.pkl', 'rb') as model:
            knn = pickle.load(model)
            st.header("KNN:")
            st.write(bool(knn.predict(data)[0]))
        with open('./models/kmeans.pkl', 'rb') as model:
            kmeans = pickle.load(model)
            st.header("KMeans:")
            st.write(bool(kmeans.predict(data)[0]))
        with open('./models/boosting.pkl', 'rb') as model:
            boosting = pickle.load(model)
            st.header("GradientBoosting:")
            st.write(bool(boosting.predict(data)[0]))
        with open('./models/bagging.pkl', 'rb') as model:
            bagging = pickle.load(model)
            st.header("Bagging:")
            st.write(bool(bagging.predict(data)[0]))
        with open('./models/stacking.pkl', 'rb') as model:
            stacking = pickle.load(model)
            st.header("Stacking:")
            st.write(bool(stacking.predict(data)[0]))
        tf_model = tf.keras.models.load_model('./models/tf.h5')
        st.header("Tensorflow:")
        st.write(bool(tf_model.predict(data)[0]))

if __name__ == "__main__":
    main()

