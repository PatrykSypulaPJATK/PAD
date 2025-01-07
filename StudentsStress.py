import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import statsmodels.formula.api as smf

# Wczytanie danych
std = pd.read_csv("student_lifestyle_dataset.csv", index_col=0)
std.dropna(inplace=True)
std['Stress_Level'] = std['Stress_Level'].apply(lambda x: 1 if x == 'Low' else 2 if x == 'Moderate' else 3)

st.header("Stres u studentów")
st.write("Cel aplikacji: Analiza danych dotyczących stylu życia studentów oraz porównanie modeli predykcyjnych.")

st.header("Eksploracja danych")
#Histogramy
if st.checkbox("Wyświetl histogramy"):
    #Histogram z suwakiem do ustawiania koszyków
    column_names = std.columns.to_list()
    selected_column_hist = st.selectbox("Select column", column_names)
    bins = st.slider("Select number of bins", min_value=5, max_value=100, value=40, step=1)
    fig, ax = plt.subplots()
    std[selected_column_hist].plot(kind="hist", bins=bins, ax=ax)
    ax.set_title(f"Histogram of {selected_column_hist.replace("_", " ").title()}")
    ax.set_xlabel(selected_column_hist.replace("_", " ").title())
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

#Zależności między zmiennymi
if st.checkbox("Wyświetl zależności między zmiennymi"):
    #Przygotowywanie selectów do wykresu
    plot_column = std.drop(columns="Stress_Level").columns.to_list()
    x_axis = st.selectbox("Select X-axis", plot_column)
    plot_column_minus = [col for col in plot_column if col != x_axis]
    y_axis = st.selectbox("Select Y-axis", plot_column_minus)
    # Tworzenie wykresu
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=x_axis,
        y=y_axis,
        hue=std.Stress_Level,
        data=std,
        palette="deep",
        ax=ax
    )
    ax.set_xlabel(x_axis.replace("_", " ").title())
    ax.set_ylabel(y_axis.replace("_", " ").title())
    ax.set_title("Scatterplot")
    st.pyplot(fig)

#Macierz korelacji
if st.checkbox("Wyświetl macierz korelacji"):
    st.subheader("Macierz korelacji")
    fig, ax = plt.subplots(figsize=(8, 6))
    tick_labels = std.columns.str.replace("_", " ").str.title()
    sns.heatmap(data=std.corr(numeric_only=True), 
        vmin=-1, vmax=1, annot=True, center=0, 
        xticklabels=tick_labels, yticklabels=tick_labels, ax=ax)
    plt.title("Macierz korelacji")
    st.pyplot(fig)


st.header("Modele predykcyjne")
# Przygotowanie danych
X = std.drop(columns="Stress_Level")
y = std["Stress_Level"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

# Backward
lm = smf.ols(formula="Stress_Level ~ Study_Hours_Per_Day + Extracurricular_Hours_Per_Day + Sleep_Hours_Per_Day + Social_Hours_Per_Day + Physical_Activity_Hours_Per_Day", data=std)
lm_bs = lm.fit()
if st.checkbox("Regresja liniowa - Backward Selection"):
    st.write(lm_bs.summary())

# Forward
lm = smf.ols(formula="Stress_Level ~ Study_Hours_Per_Day + Sleep_Hours_Per_Day", data=std)
lm_fs = lm.fit()
if st.checkbox("Regresja liniowa - Forward Selection"):
    st.write(lm_fs.summary())

# Drzewo decyzyjne
tree = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_leaf=1, min_samples_split=2, random_state=42)
tree.fit(X_train, y_train)
if st.checkbox("Drzewo decyzyjne"):
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(tree, feature_names=X.columns, filled=True, ax=ax)
    st.pyplot(fig)

# Ocena modeli
st.header("Porównanie modeli")
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    st.write(f"{model_name}:")
    st.write(f"  MSE:  {mse:.4f}")
    st.write(f"  R^2:  {r2:.4f}")

y_pred_lm_bs = lm_bs.predict(X_test)
y_pred_lm_fs = lm_fs.predict(X_test)
y_pred_tree = tree.predict(X_test)
if st.checkbox("Wyświetl porównanie"):
    evaluate_model(y_test, y_pred_lm_bs, "Regresja liniowa - Backward Selection")
    evaluate_model(y_test, y_pred_lm_fs, "Regresja liniowa - Forward Selection")
    evaluate_model(y_test, y_pred_tree, "Drzewo decyzyjne")

#Wnioski
st.header("Wnioski")
if st.checkbox("Wyświetl wnioski"):
    st.write("""
Celem aplikacji była analiza danych dotyczących stylu życia studentów oraz porównanie skuteczności różnych modeli predykcyjnych w przewidywaniu poziomu stresu.\n
Mogłoby się wydawać, że liczba godzin przeznaczonych na aktywności fizyczne oraz spędzanie czasu ze znajomymi powinny mieć duże znaczenie na ogólny stres. Aczkolwiek wyniki wskazują, że godziny snu i nauki odgrywają kluczową rolę w określaniu poziomu stresu u studentów.\n
Podsumowując studenci powinni zachowywać balans między czasem przeznaczonym na naukę a snem, aby zmniejszyć poziom stresu.
""")