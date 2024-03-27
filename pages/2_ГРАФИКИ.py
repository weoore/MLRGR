import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data= pd.read_csv("card_tr.csv")

st.title("Мошенничество с кредитными картами")
st.header("Тепловая карта")
plt.figure(figsize=(7, 5))
sns.heatmap(data.corr().round(3), annot=True, cmap='coolwarm')
st.pyplot(plt)

st.header("Круговая диаграмма")
plt.figure(figsize=(5, 5))
data['fraud'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Мошенничество')
st.pyplot(plt)

st.title('Визуализация датасета')

st.header('Датасет для классификации - "Мошенничество с картами"')

st.markdown('---')

selected_columns = st.multiselect(
    'Выберите столбцы для визуализации зависимости',
    data.columns.tolist()
)

if len(selected_columns) == 2:
    st.write(f"Гистограмма зависимости между {selected_columns[0]} и {selected_columns[1]}")
    fig, ax = plt.subplots()
    colors = ['blue', 'red']
    for i, col in enumerate(selected_columns):
        ax.scatter(data[col], data[selected_columns[1 - i]], c=colors[i], label=col)
    ax.set_xlabel(selected_columns[0])
    ax.set_ylabel(selected_columns[1])
    ax.legend()
    st.pyplot(fig)
else:
    st.warning('Пожалуйста, выберите ровно два столбца для визуализации зависимости.')

st.write("Гистограмма предсказываемого признака")

fig, ax = plt.subplots()
ax.hist(data['fraud'], bins=3)

st.pyplot(fig)
