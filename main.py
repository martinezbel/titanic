import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from utils import load_data
from utils import read_markdown_file
from utils import train_rf


def main():
    st.header("The Titanic Disaster App")
    st.markdown("Created by Oscar Martinez,  Data Scientist")
    st.markdown("(Python, Machine Learning, AI, Automation, DevOps)")
    st.subheader("Using machine learning to generate new insights into the sinking of the Titanic")
    st.markdown("This App applies data science and machine learning to analyze an existing dataset about passengers on the Titanic")
    st.sidebar.header("Analyze")

    df = load_data()
    target = "Survived"
    features = [c for c in df.columns.values if c != target]

    #description = read_markdown_file("pages/titanic.md")
    #st.image("images/titanic.jpg", width=400)
    #st.markdown(f"{description}", unsafe_allow_html=True)
    st.markdown("---")

    if st.sidebar.checkbox("Data preview", True):
        st.subheader("Data preview")
        st.markdown("The dataset analyzed is available at https://www.kaggle.com")
        st.markdown(f"This dataset contains  : {df.shape[0]} rows and, {df.shape[1]} columns")
        #if st.checkbox("Data types"):
            #st.dataframe(df.dtypes)
        if st.checkbox("Click here to see the summary statistics"):
            st.write(df.describe())
            #st.markdown("---")
            #st.markdown("---")
        st.markdown("---")
        cols_to_style = st.multiselect("Choose columns to apply BG gradient", features)
        st.dataframe(df.style.background_gradient(subset=cols_to_style, cmap="BuGn"))
        st.balloons()
        st.markdown("---")


    # if st.sidebar.checkbox("Plot distribution", False):
    #     st.subheader("Plot distribution")
    #     with st.echo():
    #         col = st.selectbox("Choose a column to display", features)
    #         with_target = st.checkbox("Separate per target ?")
    #         chart = (
    #             alt.Chart(df)
    #             .mark_bar()
    #             .encode(alt.X(f"{col}:Q", bin=alt.Bin(maxbins=10)), alt.Y("count()"),)
    #         )
    #         if with_target:
    #             chart = chart.encode(color=f"{target}:N")
    #         st.altair_chart(chart)
    #     st.markdown("---")


    if st.sidebar.checkbox("Plot distribution", False):
        st.subheader("Plot distribution")
        #st.echo():# if we add echo() displays and run  code
        col = st.selectbox("Choose column of the dataset to display in a chart", features)
        with_target = st.checkbox("Look by Survived / died")
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(alt.X(f"{col}:Q", bin=alt.Bin(maxbins=10)), alt.Y("count()"),)
        )
        if with_target:
            chart = chart.encode(color=f"{target}:N")
        st.altair_chart(chart)
    st.markdown("---")




    if st.sidebar.checkbox("Correlation", False):
        st.subheader("Correlation")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), ax=ax)
        st.pyplot(fig)
        st.markdown("---")



    if st.sidebar.checkbox("Classification", False):
        st.subheader("Aply the  Random Forest Classifier")
        st.markdown("The Random Forest algorithm creates decision trees, gets the prediction from each of them and selects the best solution. Here the algorithm predicts survived passengers ")
        n_estimators = st.number_input("Choose number of trees:", 1, 1000, 100)
        max_depth = st.number_input("Max depth:", 1, 100, 5)

        if st.button("Run training"):
            with st.spinner("Training en cours"):
                clf, confusion_matrix = train_rf(df, n_estimators, max_depth)
                st.balloons()
                st.pyplot(confusion_matrix)

        st.markdown("---")

    # st.help(pd.merge)

    st.sidebar.header("About")
    st.sidebar.text("""Created by Oscar Martinez""")
    st.sidebar.text(" Data Scientist")
    st.sidebar.text("(Python, Machine Learning, AI, DevOps)")
    #st.sidebar.text("Code : https://github.com")


if __name__ == "__main__":
    main()