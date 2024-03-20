import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

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
st.title("Ввести наблюдение")

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
    tf = tf.keras.models.load_model('./models/tf.h5')
    st.header("Tensorflow:")
    st.write(bool(tf.predict(data)[0]))
