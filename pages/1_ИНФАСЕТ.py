import streamlit as st
import pandas as pd
import numpy as np

data= pd.read_csv("card_tr.csv")
df = pd.DataFrame(data)

st.title('Информация о датасетe')

st.header('Датасет для классификации - "Мошенничество с картами"')
st.markdown('---')
st.dataframe(df)

st.subheader('distance_from_home')
st.markdown('расстояние от дома')

st.subheader('distance_from_last_transaction')
st.markdown('расстояние от последней транзакции')

st.subheader('ratio_to_median_purchase_price')
st.markdown('отношение к медианной цене покупки')

st.subheader('repeat_retailer')
st.markdown('повторное обращение к розничному продавцу')

st.subheader('used_chip')
st.markdown('использование чипа')

st.subheader('used_pin_number')
st.markdown('использование PIN-кода')

st.subheader('online_order')
st.markdown('онлайн-заказ')

st.subheader('fraud')
st.markdown('мошенничество (Если мошенничество совершенно, то 1, иначе 0 )')
