import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from urllib.parse import urlparse
from matplotlib.legend_handler import HandlerPathCollection
import missingno as msno
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from datetime import date
from statsmodels.stats.proportion import proportions_ztest
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from scipy.stats import f_oneway
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.multicomp import MultiComparison

import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt

from scipy.optimize import curve_fit  # Üstel regresyon için gerekli fonksiyon
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

from sklearn.neighbors import LocalOutlierFactor

import altair as alt

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

st.set_page_config(layout='wide', page_title='MLEBstats', page_icon='🔍')

topcol1, topcol2 = st.columns([3, 1])

html_title = """
<div style='font-size:100px; font-family:"Roboto", sans-serif; font-weight:600;'>
    <span style='color:#00ff9c;'>MLEB</span>stats 🔍
</div>
"""
topcol1.markdown(html_title, unsafe_allow_html=True)
topcol1.markdown('**Auotmated and manuel statistical analysis for professional using MLEBstats!**')
topcol2.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTs5altYHUvzqraXVwVODfVsCkw_uSVgNXWUg&s", use_column_width=True)

home_tab, overview_tab, processing_tab, post_overview_tab, modelling_tab, training_tab = st.tabs(["Home", "Pre-Overview", "Processing", "Post-Overview", "Modelling", "Train"])


col1, col2, col3 = home_tab.columns([1, 1, 1])

col1.subheader("Welcome!")
col1.markdown('Discover the power of statistics!')
col1.markdown('An application where you can perform your statistical studies in automatic and manual mode, edit and optimise your data and models according to your data set and model setup!')
col1.markdown('Who am I?')
col1.markdown('An engineer who realises the explanatory power of statistics!')
col3.subheader('What we offer?')
col3.markdown('Data preliminary analysis and visualisations, target variable analysis, hypothesis testing, data processing, advanced functional exploratory data analysis, machine learning models, prediction, time series analysis...')
col3.markdown('Analyses and reports: you will be able to output all results as reports.')
col3.markdown('Minimum requirements: basic python programming, basic statistics and basic machine learning methods.')



col2.subheader("How it works?")
col2.markdown("You need to upload the csv data. Select between automatic and manual modes. Enter information in the fields where user input is required and get the results as a report.")
col2.markdown("After entering the data set, proceed to the relevant tabs respectively.")







############################################################################################
#################### MAIN FUNCTIONS
############################################################################################

def check_df(dataframe, head=5):
    """
    Bu fonksiyon veri çerçevesini iki sütun şeklinde gösterir ve bilgileri bu sütunlarda düzenler.
    Tablo genişlik ve yüksekliği içindeki veriye göre otomatik ayarlanır.
    """



    # İki sütun oluşturma
    col1, col2 = overview_tab.columns(2)

    # Veri çerçevesinin şekli
    with col1.expander("Shape of the Dataset"):
        overview_tab.write(f"**Rows:** {dataframe.shape[0]}")
        overview_tab.write(f"**Columns/Variables:** {dataframe.shape[1]}")

    # Veri türleri (Transpoze edilmiş tablo)
    with col1.expander("Data Types"):
        dtypes_df = pd.DataFrame(dataframe.dtypes, columns=['Data Type'])
        dtypes_df.index.name = 'Column'
        overview_tab.dataframe(dtypes_df.T, use_container_width=False)  # İçeriğe göre ayarlama yapılır

    # Index bilgileri
    with col1.expander("Index Information"):
        index_info = pd.DataFrame({
            'Index Name': [dataframe.index.name if dataframe.index.name else 'None'],
            'Index Type': [type(dataframe.index).__name__],
            'Index Values': [list(dataframe.index)]
        })
        overview_tab.dataframe(index_info, use_container_width=False)  # İçeriğe göre ayarlama yapılır

    # Veri setinin genelinde eksik değer sorgusu yapma
    with col2.expander("Missing Values in Dataset (if any)"):
        has_na_values = dataframe.isnull().values.any()
        overview_tab.write(f"Any Missing Values: {'Yes' if has_na_values else 'No'}")

    # Eksik değerler (Transpoze edilmiş tablo)
    with col2.expander("The Amount of Missing Values in Each Variable (if any)"):
        missing_values_df = pd.DataFrame(dataframe.isnull().sum(), columns=['Missing Values'])
        missing_values_df.index.name = 'Column'
        overview_tab.dataframe(missing_values_df.T, use_container_width=False)  # İçeriğe göre ayarlama yapılır

    # Deskriptif istatistikler
    with col2.expander("Quantiles and Descriptive Statistics"):
        # Yüzdelik dilimlerini kullanıcıdan al
        quantiles_options = [0, 0.01, 0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99, 1]
        user_quantiles = overview_tab.multiselect(
            "Select quantiles to display:",
            options=quantiles_options,
            default=[0, 0.25, 0.50, 0.75, 1]
        )

        # Kullanıcının seçtiği veya varsayılan yüzdelik dilimlerine göre istatistikleri hesapla
        if not user_quantiles:
            user_quantiles = [0, 0.25, 0.50, 0.75, 1]  # Varsayılan yüzdelik dilimleri

        quantiles = dataframe.describe(percentiles=user_quantiles).T
        overview_tab.dataframe(quantiles, use_container_width=False)  # İçeriğe göre ayarlama yapılır


def get_column_summary(df):
    data = []
    for column in df.columns:
        total_values = df[column].dropna().shape[0]  # NA değerler hariç toplam değer sayısı
        unique_values_count = df[column].nunique()   # Benzersiz değer sayısı
        data_type = df[column].dtype                # Veri türü
        data.append({
            'Variable': column,
            'Data Type': data_type,
            'Total Values (excluding NA)': total_values,
            'Unique Values Count': unique_values_count
        })
    return pd.DataFrame(data)


def cat_summary(dataframe, col_name):
    if dataframe[col_name].dtypes == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)

    summary_df = pd.DataFrame({
        "Count": dataframe[col_name].value_counts(),
        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
    })

    # Display the summary table
    st.dataframe(summary_df)


def cat_summary_plot(dataframe, col_name, plot=True):
    if dataframe[col_name].dtypes == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)

    if plot:
        # Display the plot
        fig, ax = plt.subplots()
        sns.countplot(x=dataframe[col_name], data=dataframe, ax=ax)
        st.pyplot(fig)




def num_summary(dataframe, numerical_col):
    # Quantiles to display
    quantiles = [0, 0.01, 0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99, 1]

    # Generate summary statistics
    summary = dataframe[numerical_col].describe(quantiles).T

    # Display the summary using Streamlit
    st.write(summary)



def num_summary_plot(dataframe, numerical_col, plot=True):
    if plot:
        fig, ax = plt.subplots()
        dataframe[numerical_col].hist(ax=ax)
        ax.set_xlabel(numerical_col)
        ax.set_title(numerical_col)
        st.pyplot(fig)


def target_summary_with_cat(dataframe, target, cat_cols):
    if not cat_cols:
        st.warning(
            "Due to no categorical columns, 'The Analysis of Categorical Variables on Target Variable' couldn't be performed.")
        return

    if target not in dataframe.columns:
        st.error("Error: Target column is invalid or not selected.")
        return

    # DataFrame'in bir kopyasını oluştur, böylece orijinal DataFrame değişmez (boolean değerleri integer'a dönüştüreceğimiz için kalıcı değişiklik olmaması adına copy alıyoruz.
    df_copy = dataframe.copy()

    # Boolean değerleri integer olarak dönüştür
    for col in df_copy.columns:
        if df_copy[col].dtype == "bool":
            df_copy[col] = df_copy[col].astype(int)

    # Her satırda 2 tablo/grafik göstermek için kolonları ayarlayın
    cols_per_row = 2
    num_cols = len(cat_cols)
    num_rows = (num_cols + cols_per_row - 1) // cols_per_row

    for row in range(num_rows):
        # O satır için kolonları oluşturun
        columns = st.columns(cols_per_row * 2)  # Grafikler ve tablolara yer

        # Grafiklerin üst kısmını ayırın
        for col_index in range(cols_per_row):
            col_pos = row * cols_per_row + col_index

            if col_pos < num_cols:
                categorical_col = cat_cols[col_pos]
                with columns[col_index * 2]:
                    st.subheader(f"{categorical_col}")

                    # Gruba göre hedef değişkenin ortalamalarını hesaplayın
                    summary = pd.DataFrame({
                        "TARGET_MEAN": df_copy.groupby(categorical_col)[target].mean()
                    }).reset_index()

                    # Bar plot grafiği oluştur
                    fig, ax = plt.subplots(figsize=(6, 4))  # Grafik boyutunu ayarlayın
                    sns.barplot(x=categorical_col, y="TARGET_MEAN", data=summary, ax=ax)
                    ax.set_title(f"Mean of {target} by {categorical_col}")
                    ax.set_ylabel(f"Mean {target}")
                    ax.set_xlabel(categorical_col)
                    plt.xticks(rotation=45)  # X eksenindeki etiketleri döndür
                    st.pyplot(fig)

        # Tabloların alt kısmını ayırın
        for col_index in range(cols_per_row):
            col_pos = row * cols_per_row + col_index

            if col_pos < num_cols:
                categorical_col = cat_cols[col_pos]
                with columns[col_index * 2 + 1]:
                    st.subheader(f"{categorical_col}")

                    # Gruba göre hedef değişkenin ortalamalarını hesaplayın
                    summary = pd.DataFrame({
                        "TARGET_MEAN": df_copy.groupby(categorical_col)[target].mean()
                    })

                    # Tabloyu göster (metne göre otomatik genişletme)
                    st.dataframe(summary, use_container_width=True)  # Tabloyu tam genişlikte göster



def target_summary_with_num(dataframe, target, num_cols):
    if not num_cols:
        st.warning(
            "Due to no numerical columns, 'The Analysis of Numerical Variables on Target Variable' couldn't be performed.")
        return

    if target not in dataframe.columns:
        st.error("Error: Target column or numerical columns are invalid or not selected.")
        return

    # DataFrame'in bir kopyasını oluştur, böylece orijinal DataFrame değişmez
    df_copy = dataframe.copy()

    # Boolean değerleri integer olarak dönüştür
    for col in df_copy.columns:
        if df_copy[col].dtypes == "bool":
            df_copy[col] = df_copy[col].astype(int)

    # 'variable_tab' sekmesinde yan yana tablolara yer verecek HTML kodu
    html = "<div style='display: flex; flex-wrap: wrap;'>"

    # Her satırda 4 sütun göstermek için bir döngü oluştur
    num_cols_per_row = 4
    num_rows = (len(num_cols) + num_cols_per_row - 1) // num_cols_per_row  # Gerekli satır sayısını hesapla

    for row in range(num_rows):
        row_html = "<div style='display: flex;'>"
        for col_index in range(num_cols_per_row):
            index = row * num_cols_per_row + col_index
            if index < len(num_cols):
                numerical_col = num_cols[index]
                # Veri çerçevesini hedef değişken ve sayısal sütuna göre gruplandır ve ortalamalarını hesapla
                summary = df_copy.groupby(target).agg({numerical_col: "mean"})

                # HTML tablo kodunu oluştur
                table_html = summary.to_html(classes='dataframe', header=True, index=True)
                row_html += f"<div style='flex: 1; padding: 10px;'>{table_html}</div>"

        row_html += "</div>"
        html += row_html

    html += "</div>"

    # HTML kodunu Streamlit ile göster
    st.markdown(html, unsafe_allow_html=True)




###########################################################
###########################################################
###########################################################
###########################################################
########################################################### Buradan aşağıdaki fonksiyonlara bak . df'de kalıcı değişiklik içeren kodlar var mı?
###########################################################
###########################################################
###########################################################
###########################################################
###########################################################

# Function to calculate outlier thresholds
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# Function to check if a column has outliers
def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def grab_col_names(df, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        df: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """



    # Determine categorical columns
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and df[col].dtypes in ["int32", "float32", "int64", "float64"]]
    cat_but_car = [col for col in df.columns if df[col].nunique() > car_th and str(df[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    ### Peki biz programatik olarak nümerik değişkenleri nasıl tutarım?:
    num_cols = [col for col in df.columns if df[col].dtypes in ["int32", "int64", "float32", "float64"]]
    # categorik değişkenleri tuttuğum cat_cols listesi içinde değilse bana getir diyorum:
    num_cols = [col for col in num_cols if col not in cat_cols]

    return cat_cols, num_cols, cat_but_car, num_but_cat





def grab_outliers(dataframe, col_name, low, up, index=True):
    """
    Belirtilen sütundaki aykırı değerleri bulur ve gösterir.

    Parameters
    ------
        dataframe: dataframe
                Aykırı değerlerin kontrol edileceği veri çerçevesi
        col_name: str
                Aykırı değerlerin kontrol edileceği sütunun adı
        index: bool, optional
                Aykırı değerlerin indeks bilgilerini döndürmek için
                varsayılan False'tır

    Returns
    ------
        outlier_index: pd.Index, optional
                Eğer index=True ise, aykırı değerlerin indeks bilgilerini döndürür
    """

    outliers = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))]

    display_data = outliers


    if index:
        outlier_index = outliers.index
        return display_data, outlier_index

    return display_data







def remove_outlier(dataframe, col_name, low_limit, up_limit):
    # Fetch the existing limits set by the user in the Streamlit app

    # Filter out the outliers based on the pre-determined limits
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]

    # Display the number of outliers removed
    rows_removed = dataframe.shape[0] - df_without_outliers.shape[0]

    return df_without_outliers, rows_removed





def replace_with_thresholds(dataframe, col_name, low_limit, up_limit):
    """
    Aykırı değerleri belirtilen sınırlarla değiştirir.

    Parameters
    ------
        dataframe: pd.DataFrame
                Aykırı değerlerin değiştirileceği veri çerçevesi
        col_name: str
                Aykırı değerlerin değiştirileceği sütunun adı
        low_limit: float
                Alt sınır
        up_limit: float
                Üst sınır

    Returns
    ------
        dataframe: pd.DataFrame
                Sınırlarla değiştirilmiş veri çerçevesi
    """
    dataframe.loc[dataframe[col_name] < low_limit, col_name] = low_limit
    dataframe.loc[dataframe[col_name] > up_limit, col_name] = up_limit
    return dataframe






def update_legend_marker_size(handle, orig, size):
    "Customize size of the legend marker"
    handle.update_from(orig)
    handle.set_sizes([size])





























############################################################################################
#################### DEF MAIN (__name__ == "__main__")
############################################################################################

def main():
    st.sidebar.header("MLEBStats - Dataset Input")
    uploaded_file = st.sidebar.file_uploader("Please upload a CSV file!", type="csv", label_visibility="visible")
    st.sidebar.markdown("Please upload in csv format.")
    st.sidebar.title("Creator")
    creator_link = "https://www.linkedin.com/in/emrebaser/"
    # Creator profile link
    creator_name = "Emre Başer"
    profile_image_url = "https://media.licdn.com/dms/image/v2/C4D03AQEuq6V6H4zZTw/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1655914363315?e=1730332800&v=beta&t=pifT9Eo5AFbnEivi25wo8QylP4ZRbofRLo87rG9YcmU"  # Profil resminizin URL'si
    # Profil photo for creator
    st.sidebar.markdown(
        f"""
        <div style="display: flex; align-items: center;">
            <img src="{profile_image_url}" style="border-radius: 50%; width: 75px; height: 75px; object-fit: cover; margin-right: 10px;"/>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Creator name
    st.sidebar.header(f"{creator_name}")

    # Go to creator link!
    st.sidebar.markdown(f"[Visit the profile!]({creator_link})")


    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if df.empty:
                overview_tab.error(f"{uploaded_file} is not a valid dataset!")
            else:

                ##################################################################################################################
                ############################ PRE-OVERVIEW ÜST TABI
                ##################################################################################################################


                # Tabları oluştur
                tab1, tab2, vtab1, vtab2, vtab3, vtab4, vtab5  = overview_tab.tabs(["User Inputs","Data Structure", "Variable Analysis", "Variable Analysis on Target Variable", "Conditional Selection",
                         "Measurement Problems", "Correlation Matrix"])

                ###########################################################################
                ################ PRE-OVERVIEW ÜST TABI - User Input bölümü
                ###########################################################################

                with tab1.expander("User Input"):


                    icol1, icol2 = tab1.columns(2)

                    ################################
                    ### grab_col_names fonksiyonu için nümerik but kategoric argümanı tanımlama
                    ################################

                    # Provide options for the user
                    num_but_cat_option = icol1.radio(
                            "1.1- Choose Numeric but Categoric Argument:",
                            ("Default", "Custom")
                    )

                    # Set the value based on the user's choice
                    if num_but_cat_option == "Default":
                        num_but_cat_arg = 10  # Default value
                        # Turkuaz mavisi kutu içinde mesaj görüntüleme
                        icol1.markdown(
                            """
                            <div style="background-color: #e0f7fa; padding: 10px; border-radius: 5px; border: 1px solid #b2ebf2;">
                                <strong>1.2- The argument is set to 10 (default).</strong>
                            </div>
                            """, unsafe_allow_html=True
                        )
                    else:
                        num_but_cat_arg = icol1.number_input(
                                "1.2- Enter a custom value for Numeric but Categoric Argument:",
                                min_value=1,  # Minimum value set to 1
                                value=10,  # Initial value set to 10
                                step=1
                        )
                        # Sarı kutu içinde mesaj görüntüleme
                        icol1.markdown(
                            f"""
                            <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; border: 1px solid #ffeeba;">
                                <strong>Attention: The argument is set to: {num_but_cat_arg}. Numeric variables with unique values less than {num_but_cat_arg} will be accepted as categorical variables.</strong>
                            </div>
                            """, unsafe_allow_html=True
                        )

                    r11, r12 = icol1.columns(2)

                    # Ask the user if they want to proceed with the same value in the processing stage
                    numbutcat_process_stage = r11.radio(
                        "1.3- Do you want to continue with this value in the processing stage?",
                        ("Yes", "No")
                    )

                    if numbutcat_process_stage == "No":
                        # Sarı kutu içinde mesaj görüntüleme
                        icol1.markdown(
                            f"""
                            <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; border: 1px solid #ffeeba;">
                                <strong>Attention: Processing aşamasında her işlemde yeniden argüman girmeniz gerekecek!</strong>
                            </div>
                            """, unsafe_allow_html=True
                        )

                    # Ask the user if they want to proceed with the same value in the post-overview stage
                    numbutcat_post_overview_stage = r12.radio(
                        "1.4- Do you want to continue with this value in the post-overview stage?",
                        ("Yes", "No")
                    )

                    if numbutcat_post_overview_stage == "No":
                        # Sarı kutu içinde mesaj görüntüleme
                        icol1.markdown(
                            f"""
                            <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; border: 1px solid #ffeeba;">
                                <strong>Attention: Processing aşamasında her işlemde yeniden argüman girmeniz gerekecek!</strong>
                            </div>
                            """, unsafe_allow_html=True
                        )



                    ################################
                    ### grab_col_names fonksiyonu için categoric but cardinal argümanı tanımlama
                    ################################

                    # Provide options for the user
                    cat_but_car_option = icol2.radio(
                            "*2.1- Choose Categoric but Cardinal Argument:*",
                            ("Default", "Custom")
                    )

                    # Set the value based on the user's choice
                    if cat_but_car_option == "Default":
                            cat_but_car_arg = 20  # Default value
                            # Turkuaz mavisi kutu içinde mesaj görüntüleme
                            icol2.markdown(
                                """
                                <div style="background-color: #e0f7fa; padding: 10px; border-radius: 5px; border: 1px solid #b2ebf2;">
                                    <strong>2.2- The argument is set to 20 (default). Categoric variables with unique values bigger than 20 will be accepted as cardinal variables.</strong>
                                </div>
                                """, unsafe_allow_html=True
                            )

                    else:
                        cat_but_car_arg = icol2.number_input(
                                "2.2- Enter a custom value for Categoric but Cardinal Argument:",
                                min_value=0,
                                value=20,
                                step=1
                        )
                        # Sarı kutu içinde mesaj görüntüleme
                        icol2.markdown(
                            f"""
                            <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; border: 1px solid #ffeeba;">
                                <strong>Attention: The argument is set to: {cat_but_car_arg}. Categoric variables with unique values bigger than {cat_but_car_arg} will be accepted as cardinal variables.</strong>
                            </div>
                            """, unsafe_allow_html=True
                        )


                    r21, r22 = icol2.columns(2)


                    # Ask the user if they want to proceed with the same value in the processing stage
                    catbutcar_process_stage = r21.radio(
                        "2.3- Do you want to continue with this value in the processing stage?",
                        ("Yes", "No")
                    )

                    if catbutcar_process_stage == "No":
                        # Sarı kutu içinde mesaj görüntüleme
                        icol2.markdown(
                            f"""
                            <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; border: 1px solid #ffeeba;">
                                <strong>Attention: Processing aşamasında her işlemde yeniden argüman girmeniz gerekecek!</strong>
                            </div>
                            """, unsafe_allow_html=True
                        )

                    # Ask the user if they want to proceed with the same value in the post-overview stage
                    catbutcar_post_overview_stage = r22.radio(
                        "2.4- Do you want to continue with this value in the post-overview stage?",
                        ("Yes", "No")
                    )

                    if catbutcar_post_overview_stage == "No":
                        # Sarı kutu içinde mesaj görüntüleme
                        icol2.markdown(
                            f"""
                            <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; border: 1px solid #ffeeba;">
                                <strong>Attention: Post-Overview aşamasında her işlemde yeniden argüman girmeniz gerekecek!</strong>
                            </div>
                            """, unsafe_allow_html=True
                        )



                ###########################################################################
                ################ PRE-OVERVIEW ÜST TABI - Data Structure bölümü
                ###########################################################################


                ##########
                ocol1, ocol2 = tab2.columns([1, 1])

                #overview_tab.subheader("DataFrame Shape")

                with tab2.expander("Rows in the Dataset"):
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown("**First 5 rows**")
                        st.write(df.head())
                    with col2:
                        st.markdown("**Last 5 rows**")
                        st.write(df.tail())

                with tab2.expander("FULL Dataset"):
                    st.write(df)

                # Veri çerçevesinin şekli
                with ocol1.expander("Shape of the Dataset"):
                    st.write(f"**Rows:** {df.shape[0]}")
                    st.write(f"**Columns/Variables:** {df.shape[1]}")

                # Veri türleri (Transpoze edilmiş tablo)
                with ocol1.expander("Data Types"):
                    dtypes_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
                    dtypes_df.index.name = 'Column'
                    st.dataframe(dtypes_df.T, use_container_width=False)  # İçeriğe göre ayarlama yapılır

                # Index bilgileri
                with ocol1.expander("Index Information"):
                    index_info = pd.DataFrame({
                        'Index Name': [df.index.name if df.index.name else 'None'],
                        'Index Type': [type(df.index).__name__],
                        'Index Values': [list(df.index)]
                    })
                    st.dataframe(index_info, use_container_width=False)  # İçeriğe göre ayarlama yapılır

                # Veri setinin genelinde eksik değer sorgusu yapma
                with ocol2.expander("Missing Values in Dataset (if any)"):
                    has_na_values = df.isnull().values.any()
                    st.write(f"Any Missing Values: {'Yes' if has_na_values else 'No'}")

                # Eksik değerler (Transpoze edilmiş tablo)
                with ocol2.expander("The Amount of Missing Values in Each Variable (if any)"):
                    missing_values_df = pd.DataFrame(df.isnull().sum(), columns=['Missing Values'])
                    missing_values_df.index.name = 'Column'
                    st.dataframe(missing_values_df.T,
                                           use_container_width=False)  # İçeriğe göre ayarlama yapılır

                # Deskriptif istatistikler
                with ocol2.expander("Quantiles and Descriptive Statistics"):
                    # Yüzdelik dilimlerini kullanıcıdan al
                    quantiles_options = [0, 0.01, 0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99, 1]
                    user_quantiles = st.multiselect(
                        "Select quantiles to display:",
                        options=quantiles_options,
                        default=[0, 0.25, 0.50, 0.75, 1]
                    )

                    # Kullanıcının seçtiği veya varsayılan yüzdelik dilimlerine göre istatistikleri hesapla
                    if not user_quantiles:
                        user_quantiles = [0, 0.25, 0.50, 0.75, 1]  # Varsayılan yüzdelik dilimleri

                    quantiles = df.describe(percentiles=user_quantiles).T
                    st.dataframe(quantiles, use_container_width=False)  # İçeriğe göre ayarlama yapılır
                ##########





                with tab2:  # Overview Tab
                    with st.expander("DataFrame for Identifying the Arguments to Detect Categorical and Numerical Variables in User Inputs tab!"):

                        # Veri Çerçevesi Özeti Tablosunu Oluştur
                        df_summary = get_column_summary(df)

                        # Tabloyu Transpoze Et
                        df_summary_transposed = df_summary.T

                        # Tabloyu Streamlit ile Göster
                        st.dataframe(df_summary_transposed, use_container_width=True)



                ###########################################################################
                ################ PRE-OVERVIEW ÜST TABI - Analysis bölümü
                ###########################################################################

                with ((vtab1)):  # Diğer Tab


                    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, num_but_cat_arg, cat_but_car_arg)


                    ####################################
                    ################ PRE-OVERVIEW ÜST TABI -  Analysis bölümü - Variable Analysis Alt tabı
                    ####################################

                    with vtab1:  # Variable Analysis

                        st.button("Click for the glossary terms!", help="""#Categorical but cardinal variable:
                                It refers to a categorical variable in a dataset that has a large number of unique categories, making it somewhat similar to a continuous variable in terms of its uniqueness. These variables are technically categorical because their values represent discrete categories or groups, but they behave almost like numerical variables because the number of unique categories (or levels) is very high.
                                #Numeric but categorical variable:
                                It refers to a variable that consists of numeric values but functions as a categorical variable due to its limited number of unique values or its specific role in the analysis. Despite being stored as numeric data types, these variables represent discrete categories rather than continuous measurements.""")

                        vcol1, vcol2, vcol3 = vtab1.columns([1, 2, 3])

                        vcol1.markdown(f"Observation(s): {df.shape[0]}")
                        vcol1.markdown(f"Variable(s): {df.shape[1]}")
                        vcol2.markdown(f'Categoric but cardinal variable(s): {len(cat_but_car)}')
                        vcol2.markdown(f'Numeric but categoric variable(s): {len(num_but_cat)}')
                        vcol3.markdown(f'**Numerical variable(s):** {len(num_cols)}')
                        vcol3.markdown(
                            f'**Categoric variables (incl. numeric but categoric variable(s)):** {len(cat_cols)}')


                        ##########
                        ### PRE-OVERVIEW ÜST TABI -  Analysis bölümü - Variable Analysis Alt tabı - Analysis of Categorical Variable(s) expander
                        ##########

                        show_categorical_analysis = vtab1.checkbox("1- Show Analysis of Categorical Variables")
                        if show_categorical_analysis:


                            vtab1.header("Analysis of Categorical Variable(s)")

                            if cat_cols:  # cat_cols boş değilse
                                with vtab1:
                                    with vtab1.expander("Detected Categorical Variables:"):
                                        numcat_cols = len(cat_cols)
                                        max_cols = 6  # Maximum number of columns per row
                                        num_rows = (
                                                           numcat_cols + max_cols - 1) // max_cols  # Calculate number of rows needed

                                        for row in range(num_rows):
                                            cols = st.columns(min(max_cols,
                                                                  numcat_cols - row * max_cols))  # Create columns for the current row
                                            for i, col in enumerate(cat_cols[row * max_cols:(row + 1) * max_cols]):
                                                cols[i % len(cols)].write(
                                                    col)  # Distribute columns across the created columns
                                        else:
                                            pass

                                    with vtab1.expander("Detected Categorical but Cardinal Variables:"):
                                        numcatbutcar_cols = len(cat_but_car)
                                        max2_cols = 6  # Maximum number of columns per row
                                        num2_rows = (
                                                           numcatbutcar_cols + max2_cols - 1) // max2_cols  # Calculate number of rows needed

                                        for row in range(num2_rows):
                                            cols = st.columns(min(max2_cols,
                                                                  numcatbutcar_cols - row * max2_cols))  # Create columns for the current row
                                            for i, col in enumerate(cat_but_car[row * max2_cols:(row + 1) * max2_cols]):
                                                cols[i % len(cols)].write(
                                                    col)  # Distribute columns across the created columns
                                        else:
                                            pass

                                    with vtab1.expander("Detected Numerical but Categorical Variables:"):
                                        numnumbutcat_cols = len(num_but_cat)
                                        max3_cols = 6  # Maximum number of columns per row
                                        num3_rows = (
                                                           numnumbutcat_cols + max3_cols - 1) // max3_cols  # Calculate number of rows needed

                                        for row in range(num3_rows):
                                            cols = st.columns(min(max3_cols,
                                                                  numnumbutcat_cols - row * max3_cols))  # Create columns for the current row
                                            for i, col in enumerate(num_but_cat[row * max3_cols:(row + 1) * max3_cols]):
                                                cols[i % len(cols)].write(
                                                    col)  # Distribute columns across the created columns
                                        else:
                                            pass


                                vtab1.subheader("Table Summary")
                                # Kolon başına kaç adet sütun olacağı
                                cols_per_row = 4

                                # Kolon sayısını hesapla
                                total_cols = len(cat_cols)

                                # Kolonları satırlara bölelim
                                num_rows = (total_cols + cols_per_row - 1) // cols_per_row

                                # Her bir satıra kolonları ekleyin
                                for row in range(num_rows):
                                    # O satır için kolonları oluştur
                                    columns = vtab1.columns(cols_per_row)

                                    # Satırdaki her kolon için içerik ekleyin
                                    for col_index in range(cols_per_row):
                                        # Kolon indeksini hesaplayın
                                        col_pos = row * cols_per_row + col_index

                                        # Kolon sayısının sınırını kontrol edin
                                        if col_pos < total_cols:
                                            col_name = cat_cols[col_pos]
                                            with columns[col_index]:
                                                with st.expander(f'Summary of {col_name}'):
                                                    # Display summaries for each categorical column
                                                    cat_summary(df, col_name)

                                vtab1.subheader("Graph Summary")
                                # Kolon başına kaç adet sütun olacağı
                                cols_per_row = 4

                                # Kolon sayısını hesapla
                                total_cols = len(cat_cols)

                                # Kolonları satırlara bölelim
                                num_rows = (total_cols + cols_per_row - 1) // cols_per_row

                                # Her bir satıra kolonları ekleyin
                                for row in range(num_rows):
                                    # O satır için kolonları oluştur
                                    columns = vtab1.columns(cols_per_row)

                                    # Satırdaki her kolon için içerik ekleyin
                                    for col_index in range(cols_per_row):
                                        # Kolon indeksini hesaplayın
                                        col_pos = row * cols_per_row + col_index

                                        # Kolon sayısının sınırını kontrol edin
                                        if col_pos < total_cols:
                                            col_name = cat_cols[col_pos]

                                            # Check if the column has only NaN values
                                            if df[col_name].isna().sum() == len(df[col_name]):
                                                # If all values are NaN, display a message
                                                with columns[col_index]:
                                                    st.warning(f"The column '{col_name}' contains only NaN values.")
                                                continue  # Skip to the next column

                                            with columns[col_index]:
                                                with st.expander(f'Summary of {col_name}'):
                                                    # Display summaries for each categorical column
                                                    cat_summary_plot(df, col_name, plot=True)
                            else:
                                vtab1.write("No categorical columns to display.")

                        ##########
                        ### PRE-OVERVIEW ÜST TABI -  Analysis bölümü - Variable Analysis Alt tabı - Analysis of Numerical Variable(s) expander
                        ##########
                        show_numerical_analysis = vtab1.checkbox(" 2- Show Analysis of Numerical Variables")
                        if show_numerical_analysis:

                            vtab1.header("Analysis of Numerical Variable(s)")

                            if num_cols:  # num_cols boş değilse
                                with vtab1:
                                    with vtab1.expander("Detected Numerical Variables:"):
                                        numnum_cols = len(num_cols)
                                        max_cols = 6  # Maximum number of columns per row
                                        num_rows = (
                                                           numnum_cols + max_cols - 1) // max_cols  # Calculate number of rows needed

                                        for row in range(num_rows):
                                            cols = st.columns(min(max_cols,
                                                                  numnum_cols - row * max_cols))  # Create columns for the current row
                                            for i, col in enumerate(num_cols[row * max_cols:(row + 1) * max_cols]):
                                                cols[i % len(cols)].write(
                                                    col)  # Distribute columns across the created columns
                                        else:
                                            pass

                                vtab1.subheader("Descriptive Summary Table")
                                # Kolon başına kaç adet sütun olacağı
                                cols_per_row = 4

                                # Kolon sayısını hesapla
                                total_cols = len(num_cols)

                                # Kolonları satırlara bölelim
                                num_rows = (total_cols + cols_per_row - 1) // cols_per_row

                                # Her bir satıra kolonları ekleyin
                                for row in range(num_rows):
                                    # O satır için kolonları oluştur
                                    columns = vtab1.columns(cols_per_row)

                                    # Satırdaki her kolon için içerik ekleyin
                                    for col_index in range(cols_per_row):
                                        # Kolon indeksini hesaplayın
                                        col_pos = row * cols_per_row + col_index

                                        # Kolon sayısının sınırını kontrol edin
                                        if col_pos < total_cols:
                                            col_name = num_cols[col_pos]
                                            with columns[col_index]:
                                                with st.expander(f'Summary of {col_name}'):
                                                    # Display summaries for each categorical column
                                                    num_summary(df, col_name)

                                vtab1.subheader("Histogram Graphs")
                                # Kolon başına kaç adet sütun olacağı
                                cols_per_row = 4

                                # Kolon sayısını hesapla
                                total_cols = len(num_cols)

                                # Kolonları satırlara bölelim
                                num_rows = (total_cols + cols_per_row - 1) // cols_per_row

                                # Her bir satıra kolonları ekleyin
                                for row in range(num_rows):
                                    # O satır için kolonları oluştur
                                    columns = vtab1.columns(cols_per_row)

                                    # Satırdaki her kolon için içerik ekleyin
                                    for col_index in range(cols_per_row):
                                        # Kolon indeksini hesaplayın
                                        col_pos = row * cols_per_row + col_index

                                        # Kolon sayısının sınırını kontrol edin
                                        if col_pos < total_cols:
                                            col_name = num_cols[col_pos]
                                            with columns[col_index]:
                                                with st.expander(f'Summary of {col_name}'):
                                                    # Display summaries for each categorical column
                                                    num_summary_plot(df, col_name, plot=True)
                            else:
                                vtab1.write("No numerical columns to display.")






                    ####################################
                    ################ PRE-OVERVIEW ÜST TABI - Analysis bölümü - Target Variable Analysis alt tabı
                    ####################################
                    with vtab2:  # Target Variable Analysis

                        cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, num_but_cat_arg, cat_but_car_arg)

                        # Section to choose whether to display the categorical variable analysis
                        show_cattarget_analysis = vtab2.checkbox("Show Variable Analysis on Target Variables")

                        if show_cattarget_analysis:
                            st.session_state.target_var = df.columns[0]  # Varsayılan olarak ilk sütunu seç

                            target_var = vtab2.selectbox(
                                '*Choose a target variable*',
                                df.columns,
                                index=df.columns.get_loc(st.session_state.target_var)
                            )

                            # Seçilen hedef değişkeni güncelle
                            st.session_state.target_var = target_var

                            # Seçimi göstermek için
                            vtab2.write(f"*Selected target variable: **{st.session_state.target_var}***")

                        if show_cattarget_analysis:

                            ##########
                            ### PRE-OVERVIEW ÜST TABI -  Analysis bölümü - Variable Analysis on Target Variable Alt tabı - Function expander
                            ##########
                            with vtab2.expander("Function"):
                                st.markdown(
                                    """
                                    **Function:** 

                                    ```python
                                    #function
                                    def target_summary_with_num(dataframe, target, numerical_col):
                                        print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\\n\\n\\n")
                                    #loop
                                    for col in num_cols:
                                        target_summary_with_num(df, "survived", col)
                                    ```
                                    """
                                )


                            ##########
                            ### PRE-OVERVIEW ÜST TABI -  Analysis bölümü - Variable Analysis on Target Variable Alt tabı - The Analysis of Categorical Variables on Target Variable expander
                            ##########
                            with vtab2.expander("The Analysis of Categorical Variables on Target Variable"):
                                target_summary_with_cat(df, target_var, cat_cols)

                            ##########
                            ### PRE-OVERVIEW ÜST TABI -  Analysis bölümü - Variable Analysis on Target Variable Alt tabı - The Analysis of Numerical Variables on Target Variable expander
                            ##########
                            with vtab2.expander("The Analysis of Numerical Variables on Target Variable"):
                                target_summary_with_num(df, target_var, num_cols)




                    ####################################
                    ################ PRE-OVERVIEW ÜST TABI - Analysis bölümü - Conditional Selection alt tabı
                    ####################################
                    with vtab3:  # Target Variable Analysis
                        cacounter = 1
                        while True:
                            # Expandable section başlatma
                            with st.expander(f"Analysis {cacounter}", expanded=True):
                                # Analiz türünü seçtir
                                ca_analysis_type = st.selectbox(
                                    f"Select the type of analysis for Analysis {cacounter}",
                                    [
                                        "Hiçbir işlem yapmak istemiyorum",
                                        "Distribution Analysis",
                                        "Groupby",
                                        "Bar Plot",
                                        "Boxplot",
                                        "Violin Plot",
                                        "Histogram",
                                        "KDE Plot",
                                        "Bivariate Boxplot",
                                        "Scatter Plot",
                                        "Regression Line Plot",
                                        "Crosstab",
                                        "Frequency Table",
                                        "Stacked Bar Plot",
                                        "Pairwise Categorical Relationship",
                                        "Joint Plot",
                                        "Mosaic Plot",
                                        "Pivot Table",
                                        "Variance and Standard Deviation",
                                        "Steamgraph",
                                        "Zaman Serisi İçeren Verisetlerinde Analiz"
                                    ],
                                    key=f"ca_analysis_type_{cacounter}"
                                )

                                # Eğer "Hiçbir işlem yapmak istemiyorum" seçilirse bu döngüde işlem yapılmayacak
                                if ca_analysis_type == "Hiçbir işlem yapmak istemiyorum":
                                    st.write("Bu analizde işlem yapılmadı.")


                                ### kategorik ve nümerik columns'ların tespiti
                                cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, num_but_cat_arg,
                                                                                                  cat_but_car_arg)
                                categorical_cols = cat_cols
                                numeric_cols = num_cols

                                ###########################
                                ### Distribution Analysis
                                ###########################
                                if ca_analysis_type == "Distribution Analysis":
                                    # Kullanıcıdan sayısal ve kategorik değişkenler seçmesini iste
                                    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, num_but_cat_arg,
                                                                                                  cat_but_car_arg)
                                    categorical_cols = cat_cols
                                    numeric_cols = num_cols

                                    ca_selected_numeric = st.selectbox(
                                        f"Select a numeric column for conditional analysis (Analysis {cacounter})",
                                        options=numeric_cols,
                                        key=f"ca_selected_numeric_{cacounter}"
                                    )

                                    ca_selected_categorical = st.selectbox(
                                        f"Select a categorical column for grouping (Analysis {cacounter})",
                                        options=categorical_cols,
                                        key=f"ca_selected_categorical_{cacounter}"
                                    )

                                    # Koşul seçimi: Büyüktür, Küçüktür, veya Arasında
                                    ca_condition_type = st.radio(
                                        f"How would you like to apply the condition for the numeric column? (Analysis {cacounter})",
                                        options=["Greater than (included)", "Less than (included)", "Between (both included)"],
                                        key=f"ca_condition_type_{cacounter}"
                                    )

                                    # Kullanıcının seçimine göre threshold değerini belirle
                                    if ca_condition_type == "Greater than (included)":
                                        ca_threshold_value = st.slider(
                                            f"Select threshold for {ca_selected_numeric} (Greater than (included)) (Analysis {cacounter})",
                                            min_value=float(df[ca_selected_numeric].min()),
                                            max_value=float(df[ca_selected_numeric].max()),
                                            value=float(df[ca_selected_numeric].median()),
                                            key=f"ca_threshold_value_{cacounter}"
                                        )

                                        # Filtreleme: Büyüktür koşulu
                                        ca_filtered_df = df[df[ca_selected_numeric] >= ca_threshold_value]

                                    elif ca_condition_type == "Less than (included)":
                                        ca_threshold_value = st.slider(
                                            f"Select threshold for {ca_selected_numeric} (Less than (included)) (Analysis {cacounter})",
                                            min_value=float(df[ca_selected_numeric].min()),
                                            max_value=float(df[ca_selected_numeric].max()),
                                            value=float(df[ca_selected_numeric].median()),
                                            key=f"ca_threshold_value_{cacounter}"
                                        )

                                        # Filtreleme: Küçüktür koşulu
                                        ca_filtered_df = df[df[ca_selected_numeric] <= ca_threshold_value]

                                    elif ca_condition_type == "Between (both included)":
                                        # İki threshold değeri almak
                                        ca_threshold_min = st.slider(
                                            f"Select minimum threshold for {ca_selected_numeric} (Analysis {cacounter})",
                                            min_value=float(df[ca_selected_numeric].min()),
                                            max_value=float(df[ca_selected_numeric].max()),
                                            value=float(df[ca_selected_numeric].min()),
                                            key=f"ca_threshold_min_{cacounter}"
                                        )

                                        ca_threshold_max = st.slider(
                                            f"Select maximum threshold for {ca_selected_numeric} (Analysis {cacounter})",
                                            min_value=float(df[ca_selected_numeric].min()),
                                            max_value=float(df[ca_selected_numeric].max()),
                                            value=float(df[ca_selected_numeric].max()),
                                            key=f"ca_threshold_max_{cacounter}"
                                        )

                                        # Filtreleme: Arasında koşulu
                                        ca_filtered_df = df[(df[ca_selected_numeric] >= ca_threshold_min) & (
                                                    df[ca_selected_numeric] <= ca_threshold_max)]

                                    # Kategorik değişken için seçimler
                                    unique_categories = df[ca_selected_categorical].unique().tolist()
                                    ca_selected_categories = st.multiselect(
                                        f"Select categories for {ca_selected_categorical} (Analysis {cacounter})",
                                        options=unique_categories,
                                        default=unique_categories,
                                        key=f"ca_selected_categories_{cacounter}"
                                    )

                                    # Filtrelenmiş veriyi kategorik değişkenlere göre filtrele
                                    ca_filtered_df = ca_filtered_df[
                                        ca_filtered_df[ca_selected_categorical].isin(ca_selected_categories)]

                                    # Kullanıcının görmek istediği diğer değişkenleri seçmesini sağla
                                    ca_other_columns = st.multiselect(
                                        "Select other columns for groupby (exclude numeric column)",
                                        options=categorical_cols)

                                    if ca_other_columns:
                                        # Groupby işlemi (sayısal değişken hariç)
                                        st.subheader("Grouped Data with Counts and Percentages")
                                        grouped_data = ca_filtered_df.groupby(ca_other_columns).size().reset_index(
                                            name='Count')
                                        grouped_data['Percentage'] = (grouped_data['Count'] / grouped_data[
                                            'Count'].sum()) * 100
                                        st.write(grouped_data)
                                    else:
                                        # Filtrelenmiş veriyi göster
                                        st.subheader(f"Filtered Data for Analysis {cacounter}:")
                                        st.write(ca_filtered_df[[ca_selected_numeric, ca_selected_categorical]])

                                    # Grafiksel gösterimler
                                    st.subheader(f"Visualizations for Analysis {cacounter}")

                                    # Sayısal değişkenin dağılımı
                                    fig1, ax1 = plt.subplots(figsize=(10, 5))
                                    sns.histplot(data=ca_filtered_df, x=ca_selected_numeric, hue=ca_selected_categorical,
                                                 multiple="stack", ax=ax1)
                                    ax1.set_title(f"Distribution of {ca_selected_numeric} by {ca_selected_categorical}")
                                    st.pyplot(fig1)

                                    # Kategorik değişkene göre sayısal değişkenin ortalamasını gösteren bar grafiği
                                    st.subheader(
                                        f"Mean of {ca_selected_numeric} by {ca_selected_categorical} for Analysis {cacounter}")
                                    mean_values = ca_filtered_df.groupby(ca_selected_categorical)[
                                        ca_selected_numeric].mean().reset_index()

                                    fig2, ax2 = plt.subplots(figsize=(10, 5))
                                    sns.barplot(x=ca_selected_categorical, y=ca_selected_numeric, data=mean_values, ax=ax2)
                                    ax2.set_title(f"Mean {ca_selected_numeric} by {ca_selected_categorical}")
                                    st.pyplot(fig2)

                                    # Özet Bilgiler
                                    st.subheader(f"Summary of Filtered Data for Analysis {cacounter}")
                                    st.write(f"Number of rows after filtering: {ca_filtered_df.shape[0]}")
                                    st.write(f"Mean {ca_selected_numeric} by {ca_selected_categorical}:")
                                    st.write(mean_values)

                                ###########################
                                ### Groupby
                                ###########################
                                if ca_analysis_type == "Groupby":
                                    # Kullanıcıdan sayısal ve kategorik değişkenler seçmesini iste
                                    ca_selected_numeric = st.selectbox(
                                        f"Select a numeric column for aggregation (Analysis {cacounter})",
                                        options=numeric_cols,
                                        key=f"groupby_numeric_{cacounter}"
                                    )

                                    ca_selected_categorical = st.selectbox(
                                        f"Select a categorical column for grouping (Analysis {cacounter})",
                                        options=categorical_cols,
                                        key=f"groupby_categorical_{cacounter}"
                                    )

                                    # Kullanıcının görmek istediği diğer kategorik değişkenleri seçmesini sağla
                                    ca_other_columns = st.multiselect(
                                        "Select additional categorical columns for groupby (optional):",
                                        options=categorical_cols,
                                        key=f"groupby_other_cols_{cacounter}"
                                    )

                                    # Seçilen kategorik değişkenler ile gruplama işlemi
                                    groupby_columns = [
                                                          ca_selected_categorical] + ca_other_columns if ca_other_columns else [
                                        ca_selected_categorical]

                                    # Gruplama ve mean, median hesaplama
                                    grouped_data = df.groupby(groupby_columns)[ca_selected_numeric].agg(
                                        ['mean', 'median']).reset_index()

                                    # Sonuçları göster
                                    st.subheader(
                                        f"Grouped Data by {groupby_columns} for {ca_selected_numeric} (Analysis {cacounter})")
                                    st.write(grouped_data)


                                ###########################
                                ### Bar Plot
                                ###########################
                                elif ca_analysis_type == "Bar Plot":

                                    ### kategorik ve nümerik columns'ların tespiti
                                    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, num_but_cat_arg,
                                                                                                  cat_but_car_arg)
                                    categorical_cols = cat_cols
                                    numeric_cols = num_cols

                                    st.subheader(f"Bar Plot for Analysis {cacounter}")
                                    selected_categorical = st.selectbox(
                                        "Select a categorical column for bar plot",
                                        options=categorical_cols,
                                        key=f"bar_plot_cat_{cacounter}"
                                    )

                                    fig, ax = plt.subplots()
                                    sns.countplot(x=df[selected_categorical], ax=ax)
                                    ax.set_title(f"Bar Plot of {selected_categorical}")
                                    st.pyplot(fig)

                                ###########################
                                ### Boxplot
                                ###########################
                                elif ca_analysis_type == "Boxplot":

                                    ### kategorik ve nümerik columns'ların tespiti
                                    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, num_but_cat_arg,
                                                                                                  cat_but_car_arg)
                                    categorical_cols = cat_cols
                                    numeric_cols = num_cols

                                    st.subheader(f"Boxplot for Analysis {cacounter}")
                                    selected_categorical = st.selectbox(
                                        "Select a categorical column",
                                        options=categorical_cols,
                                        key=f"box_cat_{cacounter}"
                                    )
                                    selected_numeric = st.selectbox(
                                        "Select a numeric column",
                                        options=numeric_cols,
                                        key=f"box_num_{cacounter}"
                                    )

                                    fig, ax = plt.subplots(figsize=(10, 5))
                                    sns.boxplot(x=selected_categorical, y=selected_numeric, data=df, ax=ax)
                                    ax.set_title(f"Boxplot of {selected_numeric} by {selected_categorical}")
                                    st.pyplot(fig)

                                ###########################
                                ### Violin Plot
                                ###########################
                                elif ca_analysis_type == "Violin Plot":

                                    ### kategorik ve nümerik columns'ların tespiti
                                    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, num_but_cat_arg,
                                                                                                  cat_but_car_arg)
                                    categorical_cols = cat_cols
                                    numeric_cols = num_cols

                                    st.subheader(f"Violin Plot for Analysis {cacounter}")
                                    selected_categorical = st.selectbox(
                                        "Select a categorical column",
                                        options=categorical_cols,
                                        key=f"violin_cat_{cacounter}"
                                    )
                                    selected_numeric = st.selectbox(
                                        "Select a numeric column",
                                        options=numeric_cols,
                                        key=f"violin_num_{cacounter}"
                                    )

                                    fig, ax = plt.subplots(figsize=(10, 5))
                                    sns.violinplot(x=selected_categorical, y=selected_numeric, data=df, ax=ax)
                                    ax.set_title(f"Violin Plot of {selected_numeric} by {selected_categorical}")
                                    st.pyplot(fig)



                                ###########################
                                ### Histogram
                                ###########################
                                elif ca_analysis_type == "Histogram":

                                    ### kategorik ve nümerik columns'ların tespiti
                                    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, num_but_cat_arg,
                                                                                                  cat_but_car_arg)
                                    categorical_cols = cat_cols
                                    numeric_cols = num_cols

                                    st.subheader(f"Histogram for Analysis {cacounter}")

                                    # Kullanıcıya histogramı kategorik değişkenlerle gruplandırmak isteyip istemediğini soruyoruz
                                    group_by_category = st.checkbox("Do you want to group by a categorical variable?",
                                                                    key=f"group_by_cat_{cacounter}")

                                    # Eğer kullanıcı gruplandırmak istemezse basit histogram
                                    if not group_by_category:
                                        selected_numeric = st.selectbox(
                                            "Select a numeric column",
                                            options=numeric_cols,
                                            key=f"hist_num_{cacounter}"
                                        )

                                        fig, ax = plt.subplots()
                                        sns.histplot(df[selected_numeric], kde=False, ax=ax)
                                        ax.set_title(f"Histogram for {selected_numeric}")
                                        st.pyplot(fig)

                                    # Eğer kullanıcı gruplandırmak isterse, kategorik değişken seçtirip ona göre histogram
                                    else:
                                        selected_numeric = st.selectbox("Select a numeric column", options=numeric_cols,
                                                                        key=f"hist_numeric_{cacounter}")
                                        selected_categorical = st.selectbox("Select a categorical column",
                                                                            options=cat_cols,
                                                                            key=f"hist_categorical_{cacounter}")

                                        # 3. Gruplara Göre Nümerik Değişkenlerin Dağılımı (Histogram)
                                        st.subheader(f"3. Histogram of {selected_numeric} by {selected_categorical}")

                                        fig3, ax3 = plt.subplots(figsize=(10, 5))
                                        sns.histplot(data=df, x=selected_numeric, hue=selected_categorical,
                                                     multiple="stack", ax=ax3)
                                        ax3.set_title(f"Histogram of {selected_numeric} by {selected_categorical}")
                                        st.pyplot(fig3)




                                ###########################
                                ### KDE Plot with Separate Density and Count Axes
                                ###########################
                                elif ca_analysis_type == "KDE Plot":

                                    ### kategorik ve nümerik columns'ların tespiti
                                    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, num_but_cat_arg,
                                                                                                  cat_but_car_arg)
                                    categorical_cols = cat_cols
                                    numeric_cols = num_cols

                                    st.subheader(f"KDE Plot for Analysis {cacounter}")
                                    selected_numeric = st.selectbox(
                                        "Select a numeric column",
                                        options=numeric_cols,
                                        key=f"kde_num_{cacounter}"
                                    )

                                    # Kullanıcıya histogramı da aynı grafikte görmek isteyip istemediğini soruyoruz
                                    show_histogram = st.checkbox("Also display histogram with KDE plot?",
                                                                 key=f"show_histogram_{cacounter}")

                                    # KDE plot ve isteğe bağlı histogram ile birlikte, ayrı y eksenleri kullanarak
                                    fig, ax1 = plt.subplots()

                                    # KDE plot'u çiziyoruz (density ölçeği için sol y ekseni)
                                    sns.kdeplot(df[selected_numeric], ax=ax1, label="KDE", color="blue")
                                    ax1.set_xlabel(f"{selected_numeric}")
                                    ax1.set_ylabel("Density", color="blue")
                                    ax1.tick_params(axis="y", labelcolor="blue")

                                    if show_histogram:
                                        # İkinci y eksenini oluşturuyoruz (histogram için count ölçeği sağ y ekseni olacak)
                                        ax2 = ax1.twinx()
                                        sns.histplot(df[selected_numeric], kde=False, ax=ax2, label="Histogram",
                                                     color="orange", alpha=0.6)
                                        ax2.set_ylabel("Count", color="orange")
                                        ax2.tick_params(axis="y", labelcolor="orange")

                                    fig.suptitle(f"KDE Plot with Histogram for {selected_numeric}")
                                    ax1.legend(loc="upper left")
                                    if show_histogram:
                                        ax2.legend(loc="upper right")

                                    st.pyplot(fig)


                                ###########################
                                ### Bivariate Boxplot
                                ###########################
                                elif ca_analysis_type == "Bivariate Boxplot":

                                    ### kategorik ve nümerik columns'ların tespiti
                                    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, num_but_cat_arg,
                                                                                                  cat_but_car_arg)
                                    categorical_cols = cat_cols
                                    numeric_cols = num_cols

                                    st.subheader(f"Bivariate Boxplot for Analysis {cacounter}")
                                    selected_x = st.selectbox(
                                        "Select X-axis variable",
                                        options=numeric_cols,
                                        key=f"bivar_x_{cacounter}"
                                    )
                                    selected_y = st.selectbox(
                                        "Select Y-axis variable",
                                        options=numeric_cols,
                                        key=f"bivar_y_{cacounter}"
                                    )

                                    # Bivariate boxplot oluşturma
                                    fig, ax = plt.subplots()
                                    sns.boxplot(x=df[selected_x], y=df[selected_y], ax=ax)
                                    ax.set_title(f"Boxplot of {selected_x} vs {selected_y}")
                                    st.pyplot(fig)

                                ###########################
                                ### Scatter Plot
                                ###########################

                                if ca_analysis_type == "Scatter Plot":

                                    ### kategorik ve nümerik columns'ların tespiti
                                    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, num_but_cat_arg,
                                                                                                  cat_but_car_arg)
                                    categorical_cols = cat_cols
                                    numeric_cols = num_cols

                                    st.subheader(f"Scatter Plot for Analysis {cacounter}")

                                    # Sayısal değişken seçimi
                                    selected_x = st.selectbox("Select X-axis variable", options=numeric_cols,
                                                              key=f"scatter_x_{cacounter}")
                                    selected_y = st.selectbox("Select Y-axis variable", options=numeric_cols,
                                                              key=f"scatter_y_{cacounter}")

                                    # Kategorik değişken seçimi (isteğe bağlı)
                                    selected_categorical = st.selectbox(
                                        "Select a categorical variable for grouping (optional)",
                                        options=[None] + cat_cols,
                                        key=f"scatter_categorical_{cacounter}"
                                    )

                                    # Kullanıcıya regresyon çizgisi eklemek isteyip istemediğini soruyoruz
                                    add_regression_line = st.checkbox("Add regression line?",
                                                                      key=f"regression_line_{cacounter}")

                                    if add_regression_line:
                                        # Regresyon türünü seçtiriyoruz
                                        regression_type = st.selectbox(
                                            "Select the type of regression line",
                                            ["Linear", "Polynomial", "Logarithmic", "Exponential"],
                                            key=f"regression_type_{cacounter}"
                                        )

                                        # Eğer polinomial regresyon seçildiyse, derecesini kullanıcıya seçtiriyoruz
                                        if regression_type == "Polynomial":
                                            poly_degree = st.slider(
                                                "Select the degree of the polynomial",
                                                min_value=2,
                                                max_value=10,
                                                value=2,
                                                key=f"poly_degree_{cacounter}"
                                            )

                                        # Kullanıcı kategorik değişken bazlı regresyon çizgisi isteyip istemediğini seçebilir
                                        category_based_line = st.checkbox(
                                            "Draw regression line for each category (if categorical variable is selected)?",
                                            key=f"category_based_line_{cacounter}"
                                        )

                                    # Scatter plot oluşturma
                                    fig, ax = plt.subplots(figsize=(10, 5))

                                    if selected_categorical:
                                        scatter = sns.scatterplot(x=df[selected_x], y=df[selected_y],
                                                                  hue=df[selected_categorical], ax=ax)
                                        # Renk paletini elde ediyoruz
                                        palette = scatter.legend_.legendHandles
                                    else:
                                        sns.scatterplot(x=df[selected_x], y=df[selected_y], ax=ax)

                                    # Eğer regresyon çizgisi eklenmek isteniyorsa
                                    if add_regression_line:
                                        if selected_categorical and category_based_line:
                                            # Her kategori için ayrı bir regresyon çizgisi çizme
                                            unique_categories = df[selected_categorical].unique()
                                            for idx, category in enumerate(unique_categories):
                                                category_data = df[df[selected_categorical] == category]
                                                # Renk paletinden kategorinin rengini seçiyoruz
                                                category_color = palette[idx].get_facecolor()[:3]
                                                if regression_type == "Linear":
                                                    sns.regplot(x=category_data[selected_x],
                                                                y=category_data[selected_y], ax=ax, scatter=False,
                                                                label=f"{category} Regression",
                                                                line_kws={"color": category_color})
                                                elif regression_type == "Polynomial":
                                                    sns.regplot(x=category_data[selected_x],
                                                                y=category_data[selected_y], ax=ax, scatter=False,
                                                                order=poly_degree, label=f"{category} Polynomial",
                                                                line_kws={"color": category_color})
                                                elif regression_type == "Logarithmic":
                                                    sns.regplot(x=category_data[selected_x],
                                                                y=category_data[selected_y], ax=ax, scatter=False,
                                                                logx=True, label=f"{category} Logarithmic",
                                                                line_kws={"color": category_color})
                                                elif regression_type == "Exponential":
                                                    # Exponential regresyon için kendi modeli oluşturuyoruz
                                                    def exp_func(x, a, b):
                                                        return a * np.exp(b * x)

                                                    x_data = category_data[selected_x].values
                                                    y_data = category_data[selected_y].values
                                                    x_norm = (x_data - np.min(x_data)) / (
                                                                np.max(x_data) - np.min(x_data))
                                                    y_norm = (y_data - np.min(y_data)) / (
                                                                np.max(y_data) - np.min(y_data))

                                                    try:
                                                        popt, _ = curve_fit(exp_func, x_norm, y_norm, p0=(1, 1),
                                                                            maxfev=5000)
                                                        x_range = np.linspace(np.min(x_norm), np.max(x_norm), 100)
                                                        y_range = exp_func(x_range, *popt)
                                                        ax.plot(x_data,
                                                                y_range * (np.max(y_data) - np.min(y_data)) + np.min(
                                                                    y_data),
                                                                color=category_color, label=f"{category} Exponential")
                                                    except RuntimeError:
                                                        st.write(f"Exponential fit could not be found for {category}")

                                        else:
                                            # Tek regresyon çizgisi
                                            if regression_type == "Linear":
                                                sns.regplot(x=df[selected_x], y=df[selected_y], ax=ax, scatter=False,
                                                            line_kws={"color": "red"})
                                            elif regression_type == "Polynomial":
                                                sns.regplot(x=df[selected_x], y=df[selected_y], ax=ax, scatter=False,
                                                            order=poly_degree,
                                                            line_kws={"color": "green"})
                                            elif regression_type == "Logarithmic":
                                                sns.regplot(x=df[selected_x], y=df[selected_y], ax=ax, scatter=False,
                                                            logx=True,
                                                            line_kws={"color": "blue"})
                                            elif regression_type == "Exponential":
                                                def exp_func(x, a, b):
                                                    return a * np.exp(b * x)

                                                x_data = df[selected_x].values
                                                y_data = df[selected_y].values
                                                x_norm = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
                                                y_norm = (y_data - np.min(y_data)) / (np.max(y_data) - np.min(y_data))

                                                try:
                                                    popt, _ = curve_fit(exp_func, x_norm, y_norm, p0=(1, 1),
                                                                        maxfev=5000)
                                                    x_range = np.linspace(np.min(x_norm), np.max(x_norm), 100)
                                                    y_range = exp_func(x_range, *popt)
                                                    ax.plot(x_data,
                                                            y_range * (np.max(y_data) - np.min(y_data)) + np.min(
                                                                y_data), color="purple")
                                                except RuntimeError:
                                                    st.write("Exponential fit could not be found for the data.")

                                    ax.set_title(f"Scatter Plot of {selected_x} vs {selected_y}")
                                    ax.legend()
                                    st.pyplot(fig)



                                ###########################
                                ### Regression Line Plot
                                ###########################
                                elif ca_analysis_type == "Regression Line Plot":

                                    ### kategorik ve nümerik columns'ların tespiti
                                    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, num_but_cat_arg,
                                                                                                  cat_but_car_arg)
                                    categorical_cols = cat_cols
                                    numeric_cols = num_cols


                                    st.subheader(f"Scatter Plot with Regression Line for Analysis {cacounter}")
                                    selected_x = st.selectbox(
                                        "Select X-axis variable",
                                        options=numeric_cols,
                                        key=f"reg_x_{cacounter}"
                                    )
                                    selected_y = st.selectbox(
                                        "Select Y-axis variable",
                                        options=numeric_cols,
                                        key=f"reg_y_{cacounter}"
                                    )

                                    fig, ax = plt.subplots()
                                    sns.regplot(x=df[selected_x], y=df[selected_y], ax=ax)
                                    ax.set_title(f"Regression Plot of {selected_x} vs {selected_y}")
                                    st.pyplot(fig)

                                ###########################
                                ### Crosstab Analysis
                                ###########################
                                elif ca_analysis_type == "Crosstab":

                                    ### kategorik ve nümerik columns'ların tespiti
                                    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, num_but_cat_arg,
                                                                                                  cat_but_car_arg)


                                    st.subheader(f"Crosstab for Analysis {cacounter}")

                                    # Kategorik değişkenleri seçtir
                                    selected_cat1 = st.selectbox(
                                        "Select the first categorical variable",
                                        options=cat_cols,
                                        key=f"crosstab_cat1_{cacounter}"
                                    )
                                    selected_cat2 = st.selectbox(
                                        "Select the second categorical variable",
                                        options=cat_cols,
                                        key=f"crosstab_cat2_{cacounter}"
                                    )

                                    # Crosstab oluşturma
                                    crosstab = pd.crosstab(df[selected_cat1], df[selected_cat2])
                                    crosstab_percent = pd.crosstab(df[selected_cat1], df[selected_cat2],
                                                                   normalize='index') * 100

                                    # Crosstab gösterimi
                                    st.write("Crosstab:")
                                    st.write(crosstab)

                                    # Yüzde şeklinde normalize edilmiş crosstab
                                    st.write("Crosstab as percentage:")
                                    st.write(crosstab_percent)

                                ###########################
                                ### Frequency Table
                                ###########################
                                elif ca_analysis_type == "Frequency Table":

                                    ### kategorik ve nümerik columns'ların tespiti
                                    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, num_but_cat_arg,
                                                                                                  cat_but_car_arg)
                                    categorical_cols = cat_cols
                                    numeric_cols = num_cols


                                    st.subheader(f"Frequency Table for Analysis {cacounter}")
                                    selected_cat = st.selectbox(
                                        "Select a categorical variable",
                                        options=categorical_cols,
                                        key=f"freq_cat_{cacounter}"
                                    )

                                    freq_table = df[selected_cat].value_counts()
                                    st.write(
                                        pd.DataFrame({"Count": freq_table, "Percentage": freq_table / len(df) * 100}))

                                ###########################
                                ### Stacked Bar Plot
                                ###########################
                                elif ca_analysis_type == "Stacked Bar Plot":

                                    ### kategorik ve nümerik columns'ların tespiti
                                    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, num_but_cat_arg,
                                                                                                  cat_but_car_arg)
                                    categorical_cols = cat_cols
                                    numeric_cols = num_cols

                                    st.subheader(f"Stacked Bar Plot for Analysis {cacounter}")
                                    selected_cat1 = st.selectbox(
                                        "Select the first categorical variable",
                                        options=categorical_cols,
                                        key=f"stacked_cat1_{cacounter}"
                                    )
                                    selected_cat2 = st.selectbox(
                                        "Select the second categorical variable",
                                        options=categorical_cols,
                                        key=f"stacked_cat2_{cacounter}"
                                    )

                                    crosstab = pd.crosstab(df[selected_cat1], df[selected_cat2])

                                    fig, ax = plt.subplots()
                                    crosstab.plot(kind="bar", stacked=True, ax=ax)
                                    ax.set_title(f"Stacked Bar Plot of {selected_cat1} by {selected_cat2}")
                                    st.pyplot(fig)

                                ###########################
                                ### Pairwise Categorical Relationship
                                ###########################
                                elif ca_analysis_type == "Pairwise Categorical Relationship":

                                    ### kategorik ve nümerik columns'ların tespiti
                                    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, num_but_cat_arg,
                                                                                                  cat_but_car_arg)
                                    categorical_cols = cat_cols
                                    numeric_cols = num_cols

                                    st.subheader(f"Pairwise Categorical Relationship for Analysis {cacounter}")
                                    selected_cat1 = st.selectbox(
                                        "Select the first categorical variable",
                                        options=categorical_cols,
                                        key=f"pairwise_cat1_{cacounter}"
                                    )
                                    selected_cat2 = st.selectbox(
                                        "Select the second categorical variable",
                                        options=categorical_cols,
                                        key=f"pairwise_cat2_{cacounter}"
                                    )

                                    fig, ax = plt.subplots()
                                    sns.countplot(x=df[selected_cat1], hue=df[selected_cat2], ax=ax)
                                    ax.set_title(f"Count Plot of {selected_cat1} by {selected_cat2}")
                                    st.pyplot(fig)

                                ###########################
                                ### Joint Plot
                                ###########################
                                elif ca_analysis_type == "Joint Plot":

                                    ### kategorik ve nümerik columns'ların tespiti
                                    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, num_but_cat_arg,
                                                                                                  cat_but_car_arg)
                                    categorical_cols = cat_cols
                                    numeric_cols = num_cols
                                    all_cols = num_cols + cat_cols

                                    st.subheader(f"Joint Plot for Analysis {cacounter}")
                                    selected_x = st.selectbox(
                                        "Select X-axis variable",
                                        options=all_cols,
                                        key=f"joint_x_{cacounter}"
                                    )
                                    selected_y = st.selectbox(
                                        "Select Y-axis variable",
                                        options=all_cols,
                                        key=f"joint_y_{cacounter}"
                                    )

                                    # Jointplot oluşturma
                                    fig = sns.jointplot(x=selected_x, y=selected_y, data=df, kind="scatter")
                                    st.pyplot(fig)
                                ###########################
                                ### Mosaic Plot
                                ###########################
                                elif ca_analysis_type == "Mosaic Plot":

                                    ### kategorik ve nümerik columns'ların tespiti
                                    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, num_but_cat_arg,
                                                                                                  cat_but_car_arg)
                                    categorical_cols = cat_cols
                                    numeric_cols = num_cols
                                    all_cols = num_cols + cat_cols

                                    from statsmodels.graphics.mosaicplot import mosaic
                                    from itertools import cycle
                                    import matplotlib.colors as mcolors
                                    import matplotlib.patches as mpatches

                                    st.subheader(f"Mosaic Plot for Analysis {cacounter}")

                                    selected_cat1 = st.selectbox(
                                        "Select the first categorical variable",
                                        options=categorical_cols,
                                        key=f"mosaic_cat1_{cacounter}"
                                    )
                                    selected_cat2 = st.selectbox(
                                        "Select the second categorical variable",
                                        options=[col for col in categorical_cols if col != selected_cat1],
                                        key=f"mosaic_cat2_{cacounter}"
                                    )

                                    # Profesyonel gözükmesi için pastel tonlarda renk paleti
                                    pastel_colors = list(
                                        mcolors.TABLEAU_COLORS.values())  # Tableau'nun profesyonel renk paleti

                                    # selected_cat2'nin unique değerlerini elde ediyoruz ve string'e çeviriyoruz
                                    unique_values_cat2 = df[selected_cat2].astype(str).unique()

                                    # selected_cat2'nin her unique değeri için bir renk atıyoruz
                                    color_dict = {value: color for value, color in
                                                  zip(unique_values_cat2, cycle(pastel_colors))}

                                    # Özelleştirilmiş renklerle mosaic plot
                                    def custom_mosaic_props(key):
                                        # key[1] selected_cat2'yi ifade eder, bu değere göre renk belirliyoruz
                                        return {'color': color_dict[str(key[1])]}  # key[1] string olarak işleniyor

                                    # labelizer fonksiyonunu gruplama işlemini düzeltmek için yeniden yazıyoruz
                                    def labelizer(key):
                                        key_tuple = tuple(key)
                                        group_size = df.groupby([selected_cat1, selected_cat2]).size()

                                        if key_tuple in group_size.index:
                                            value = group_size[key_tuple]
                                            percentage = value / len(df) * 100
                                            return f'{key[0]} & {key[1]}\n({percentage:.1f}%)'
                                        else:
                                            return ''

                                    fig, ax = plt.subplots(figsize=(10, 8))

                                    # Mosaic plot oluşturma
                                    mosaic(df, [selected_cat1, selected_cat2], ax=ax, properties=custom_mosaic_props,
                                           labelizer=labelizer)

                                    # Eksenin solundaki değerleri kaldırma
                                    ax.get_yaxis().set_visible(False)

                                    # Renklerin hangi sınıfa ait olduğunu göstermek için legend ekleme
                                    legend_patches = [
                                        mpatches.Patch(color=color_dict[val], label=f'{selected_cat2}: {val}') for val
                                        in unique_values_cat2]
                                    ax.legend(handles=legend_patches, title="Groups", bbox_to_anchor=(1.05, 1),
                                              loc='upper left')

                                    ax.set_title(f"Mosaic Plot of {selected_cat1} and {selected_cat2}", fontsize=14)
                                    st.pyplot(fig)


                                ###########################
                                ### Pivot Table
                                ###########################
                                elif ca_analysis_type == "Pivot Table":

                                    ### kategorik ve nümerik columns'ların tespiti
                                    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, num_but_cat_arg,
                                                                                                  cat_but_car_arg)
                                    categorical_cols = cat_cols
                                    numeric_cols = num_cols

                                    st.subheader(f"Pivot Table for Analysis {cacounter}")
                                    selected_numeric = st.selectbox(
                                        "Select a numeric column",
                                        options=numeric_cols,
                                        key=f"pivot_num_{cacounter}"
                                    )
                                    selected_cat = st.selectbox(
                                        "Select a categorical column",
                                        options=categorical_cols,
                                        key=f"pivot_cat_{cacounter}"
                                    )

                                    pivot_table = pd.pivot_table(df, values=selected_numeric, index=selected_cat,
                                                                 aggfunc=['mean', 'sum'])
                                    st.write(pivot_table)

                                ###########################
                                ### Variance and Standard Deviation
                                ###########################
                                elif ca_analysis_type == "Variance and Standard Deviation":

                                    ### kategorik ve nümerik columns'ların tespiti
                                    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, num_but_cat_arg,
                                                                                                  cat_but_car_arg)
                                    categorical_cols = cat_cols
                                    numeric_cols = num_cols
                                    all_cols = num_cols + cat_cols


                                    st.subheader(f"Variance and Standard Deviation for Analysis {cacounter}")
                                    selected_numeric = st.selectbox(
                                        "Select a numeric column",
                                        options=numeric_cols,
                                        key=f"var_num_{cacounter}"
                                    )
                                    selected_cat = st.selectbox(
                                        "Select a categorical column",
                                        options=categorical_cols,
                                        key=f"var_cat_{cacounter}"
                                    )

                                    grouped_var = df.groupby(selected_cat)[selected_numeric].var().reset_index()
                                    grouped_std = df.groupby(selected_cat)[selected_numeric].std().reset_index()

                                    st.write(f"Variance of {selected_numeric} by {selected_cat}:")
                                    st.write(grouped_var)
                                    st.write(f"Standard Deviation of {selected_numeric} by {selected_cat}:")
                                    st.write(grouped_std)

                                ###########################
                                ### Zaman Serisi İçeren Verisetlerinde Analiz
                                ###########################
                                if ca_analysis_type == "Zaman Serisi İçeren Verisetlerinde Analiz":

                                    st.subheader(
                                        "zaman serisi konusuna gelindiğinde tekrar gözden geçir!!!!! saçma sapan bir kurgu yaptım :D ")

                                    ### kategorik ve nümerik columns'ların tespiti
                                    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, num_but_cat_arg,
                                                                                                  cat_but_car_arg)
                                    categorical_cols = cat_cols
                                    numeric_cols = num_cols
                                    all_cols = num_cols + cat_cols

                                    an_df = df.copy()

                                    ##############################
                                    #### STRUCTURE - Time Variable Processing
                                    ##############################
                                    time_var_present = st.radio(
                                        "Is there a time variable in the DataFrame?",
                                        ("No", "Yes"),
                                        key=f"time_var_present_{cacounter}"
                                    )

                                    if time_var_present == "Yes":
                                        # Zaman değişkenini seçelim
                                        selected_time_var = st.selectbox(
                                            "Select the time variable:",
                                            an_df.columns,
                                            key=f"time_var_select_{cacounter}"
                                        )

                                        # Zaman değişkenini datetime formata çevirelim
                                        if not pd.api.types.is_datetime64_any_dtype(an_df[selected_time_var]):
                                            convert_to_datetime = st.radio(
                                                f"'{selected_time_var}' is not a datetime type. Do you want to convert it?",
                                                ("No", "Yes"),
                                                key=f"convert_to_datetime_{cacounter}"
                                            )
                                            if convert_to_datetime == "Yes":
                                                an_df[selected_time_var] = pd.to_datetime(an_df[selected_time_var], utc=True)
                                                st.write(
                                                    f"'{selected_time_var}' has been successfully converted to datetime.")

                                        ##############################
                                        #### Zaman Serisi Analiz Türü Seçimi
                                        ##############################
                                        analysis_type = st.radio(
                                            "Select the type of time series analysis you want to perform:",
                                            [
                                                "Time Series Visualization",
                                                "Decomposition Analysis"
                                            ],
                                            key=f"analysis_type_{cacounter}"
                                        )

                                        ##############################
                                        #### Visualization İşlemi
                                        ##############################
                                        if analysis_type == "Time Series Visualization":
                                            st.subheader("Time Series Visualization")

                                            numeric_cols = df.select_dtypes(
                                                include=['float64', 'int64']).columns.tolist()
                                            target_col = st.selectbox(
                                                "Select the target variable (numeric):",
                                                options=numeric_cols,
                                                key=f"target_col_{cacounter}"
                                            )

                                            categorical_cols = df.select_dtypes(
                                                include=['object', 'category']).columns.tolist()
                                            selected_categorical = st.multiselect(
                                                "Select categorical variables for filtering or groupby (optional)",
                                                options=categorical_cols
                                            )

                                            group_or_filter = st.radio(
                                                "Do you want to filter or group by the selected categorical variables?",
                                                ("Filter", "Groupby"),
                                                key=f"group_or_filter_{cacounter}"
                                            )

                                            if group_or_filter == "Groupby" and selected_categorical:
                                                fig, ax = plt.subplots(figsize=(10, 5))
                                                an_df_grouped = an_df.groupby(
                                                    [selected_time_var] + selected_categorical).mean().reset_index()
                                                sns.lineplot(x=an_df_grouped[selected_time_var],
                                                             y=an_df_grouped[target_col],
                                                             hue=an_df_grouped[selected_categorical[0]], ax=ax)
                                                ax.set_title(
                                                    f"Time Series Plot of {target_col} grouped by {selected_categorical[0]}")
                                                ax.set_xlabel("Date")
                                                ax.set_ylabel(target_col)
                                                st.pyplot(fig)
                                            else:
                                                fig, ax = plt.subplots(figsize=(10, 5))
                                                ax.plot(an_df[selected_time_var], an_df[target_col], label=target_col)
                                                ax.set_title(f"Time Series Plot of {target_col}")
                                                ax.set_xlabel("Date")
                                                ax.set_ylabel(target_col)
                                                st.pyplot(fig)

                                        ##############################
                                        #### Decomposition Analysis İşlemi
                                        ##############################
                                        elif analysis_type == "Decomposition Analysis":
                                            st.subheader("Decomposition Analysis")

                                            numeric_cols = df.select_dtypes(
                                                include=['float64', 'int64']).columns.tolist()
                                            target_col = st.selectbox(
                                                "Select the target variable (numeric):",
                                                options=numeric_cols,
                                                key=f"target_col_{cacounter}"
                                            )

                                            # Frekans seçimi: Kullanıcıya frekansı manuel seçme veya otomatik tahmin seçeneği sunuyoruz
                                            define_freq = st.radio(
                                                "Do you want to define a frequency for the time variable?",
                                                ("Auto Detect", "Manually Select"),
                                                key=f"define_freq_{cacounter}"
                                            )

                                            # Frekansı auto detect ile bulma
                                            if define_freq == "Auto Detect":
                                                freq_code = pd.infer_freq(an_df[selected_time_var])
                                                if freq_code:
                                                    st.write(f"Auto Detected Frequency: {freq_code}")
                                                else:
                                                    st.error(
                                                        "Frequency could not be inferred. Please manually select a frequency.")
                                                    freq_code = None

                                            # Kullanıcı manuel seçerse
                                            elif define_freq == "Manually Select":
                                                freq_options = {
                                                    'Hourly': 'H',
                                                    'Minutely': 'T',
                                                    'Secondly': 'S',
                                                    'Daily': 'D',
                                                    'Business Day': 'B',
                                                    'Weekly': 'W',
                                                    'Monthly': 'M',
                                                    'Quarterly': 'Q',
                                                    'Yearly': 'Y'
                                                }
                                                selected_freq = st.selectbox(
                                                    "Select the frequency for the time variable:",
                                                    list(freq_options.keys()),
                                                    key=f"freq_select_{cacounter}"
                                                )
                                                freq_code = freq_options[selected_freq]

                                            # Zaman değişkenini indexe alıp frekansı ayarlıyoruz
                                            if freq_code:
                                                an_df = an_df.set_index(
                                                    pd.DatetimeIndex(an_df[selected_time_var])).asfreq(freq_code)

                                                # Eksik verileri nasıl ele alacağımızı kullanıcıya soralım
                                                handle_missing_values = st.radio(
                                                    "How do you want to handle missing values in the time series?",
                                                    ("Remove Missing Values",
                                                     "Fill Missing Values with Forward Fill",
                                                     "Fill Missing Values with Backward Fill",
                                                     "Fill Missing Values with Zero",
                                                     "Fill Missing Values with Mean",
                                                     "Interpolate Missing Values"),
                                                    key=f"handle_missing_{cacounter}"
                                                )

                                                if handle_missing_values == "Remove Missing Values":
                                                    an_df = an_df.dropna(subset=[target_col])
                                                elif handle_missing_values == "Fill Missing Values with Forward Fill":
                                                    an_df[target_col] = an_df[target_col].fillna(method='ffill')
                                                elif handle_missing_values == "Fill Missing Values with Backward Fill":
                                                    an_df[target_col] = an_df[target_col].fillna(method='bfill')
                                                elif handle_missing_values == "Fill Missing Values with Zero":
                                                    an_df[target_col] = an_df[target_col].fillna(0)
                                                elif handle_missing_values == "Fill Missing Values with Mean":
                                                    mean_value = an_df[target_col].mean()
                                                    an_df[target_col] = an_df[target_col].fillna(mean_value)
                                                elif handle_missing_values == "Interpolate Missing Values":
                                                    an_df[target_col] = an_df[target_col].interpolate(method='linear')

                                                # İşlem sonrası yeni DataFrame gösteriliyor
                                                st.write("Updated DataFrame after handling missing values:")
                                                st.write(an_df)

                                                # Resample yapma seçeneği
                                                resample_data = st.radio(
                                                    "Do you want to resample the time series data?",
                                                    ("No", "Yes"),
                                                    key=f"resample_{cacounter}"
                                                )

                                                if resample_data == "Yes":
                                                    resample_freq = st.selectbox(
                                                        "Select the resampling frequency:",
                                                        ['MS', 'Q', 'Y'],  # Monthly, Quarterly, Yearly
                                                        key=f"resample_freq_{cacounter}"
                                                    )
                                                    an_df = an_df.resample(resample_freq).mean()

                                                # Model seçimi
                                                decomposition_model = st.selectbox(
                                                    "Select the decomposition model (Additive or Multiplicative):",
                                                    ["additive", "multiplicative"],
                                                    key=f"decomposition_model_{cacounter}"
                                                )

                                                # Durağanlık testi isteği
                                                stationary_test = st.checkbox("Perform stationarity test (ADF)?",
                                                                              key=f"stationarity_test_{cacounter}")

                                                # Decomposition ve Stationarity Test Fonksiyonları
                                                def is_stationary(y):
                                                    p_value = sm.tsa.stattools.adfuller(y)[1]
                                                    if p_value < 0.05:
                                                        st.write(
                                                            f"Result: Stationary (H0: non-stationary rejected, p-value: {p_value:.4f})")
                                                    else:
                                                        st.write(
                                                            f"Result: Non-Stationary (H0: non-stationary accepted, p-value: {p_value:.4f})")

                                                # Resample fonksiyonunu decomposition ile birleştirme
                                                def ts_decompose(y, model="additive", stationary=False,
                                                                 resample_freq=None):
                                                    y.index = pd.to_datetime(y.index)

                                                    if resample_freq:
                                                        y = y.resample(resample_freq).mean()
                                                        y.dropna(inplace=True)

                                                    result = seasonal_decompose(y, model=model)

                                                    fig, axes = plt.subplots(4, 1, sharex=True, sharey=False)
                                                    fig.set_figheight(10)
                                                    fig.set_figwidth(15)

                                                    axes[0].set_title("Decomposition for " + model + " model")
                                                    axes[0].plot(y, 'k', label='Original ' + model)
                                                    axes[0].legend(loc='upper left')

                                                    axes[1].plot(result.trend, label='Trend')
                                                    axes[1].legend(loc='upper left')

                                                    axes[2].plot(result.seasonal, 'g',
                                                                 label='Seasonality & Mean: ' + str(
                                                                     round(result.seasonal.mean(), 4)))
                                                    axes[2].legend(loc='upper left')

                                                    axes[3].plot(result.resid, 'r', label='Residuals & Mean: ' + str(
                                                        round(result.resid.mean(), 4)))
                                                    axes[3].legend(loc='upper left')
                                                    st.pyplot(fig)

                                                    if stationary:
                                                        is_stationary(y)

                                                # Decomposition işlemi
                                                ts_decompose(an_df[target_col], model=decomposition_model,
                                                             stationary=stationary_test, resample_freq=resample_freq)



                                if ca_analysis_type == "Steamgraph":
                                    st.write("Steamgraph eklenecek")

                                # Kullanıcıya başka bir işlem yapmak isteyip istemediği soruluyor
                                ca_continue_or_not = st.radio(
                                    "Başka bir işlem yapmak istiyor musunuz?",
                                    ("Evet", "Hayır"),
                                    index=1,
                                    key=f"ca_continue_or_not_{cacounter}"
                                )

                                if ca_continue_or_not == "Hayır":
                                    st.write("Döngü sonlandırıldı.")
                                    break

                            # Her adımda cacounter artırılıyor
                            cacounter += 1

                    ####################################
                    ################ PRE-OVERVIEW ÜST TABI - Analysis bölümü - Measurement Problems alt tabı
                    ####################################

                    mp_feature_options = [
                        "Hiçbir işlem yapmak istemiyorum",  # Varsayılan seçenek
                        "İki Örneklem T Testi",
                        "İki Örneklem Oran Testi",
                        "ANOVA"
                    ]

                    mpcounter = 1
                    while True:
                        with vtab4.expander(f"İşlem Seçimi {mpcounter}", expanded=True):
                            mpcol1, mpcol2 = st.columns(2)

                            # Kullanıcıya önce kategori seçimi sunuluyor
                            mp_category_choice = mpcol1.selectbox(
                                "Lütfen işlem kategorisini seçin:",
                                options=mp_feature_options,
                                key=f"xmp_selectbox_category_{mpcounter}"  # mp_counter ile benzersiz key
                            )

                            if mp_category_choice == "Hiçbir işlem yapmak istemiyorum":
                                break

                            ################################################################################################
                            ##################   İki Örneklem T Testi
                            ################################################################################################

                            # Eğer "İki Örneklem T Testi" seçilirse
                            if mp_category_choice == "İki Örneklem T Testi":

                                st.button("Yorumlama hk.", help="""Soru: iki örneklem T testi sonucunda anlamlı ilişki bulurken iki örneklem oran testinde anlamlı ilişki yoktur sonucuna ulaşıyorum. bu mümkün müdür? nasıl yorumlamam gerekiyor? Cevap: Evet, iki örneklem T testi ve iki örneklem oran testi farklı sonuçlar verebilir, ve bu durumu anlamak ve yorumlamak önemlidir. Bunun mümkün olup olmadığını ve nasıl yorumlanacağını daha ayrıntılı olarak açıklayayım:
    
                                ### 1. **İki Örneklem T Testi**
                                - **Amaç**: İki örneklem T testi, iki grup arasındaki **ortalama farkını** test eder. Yani, iki grubun ortalamaları arasında anlamlı bir fark olup olmadığını kontrol eder.
                                - **Varsayımlar**: 
                                  - Verilerin normal dağılıma sahip olması.
                                  - Varyansların homojen olması (eşit varyans).
                                  - Bağımsız örneklem.
    
                                T testi, sayısal (kesintisiz) bir değişkenin iki grup arasında ortalamasını karşılaştırmak için kullanılır. Eğer gruplar arasında ortalama farkı önemliyse, bu test sonucunda anlamlı bir ilişki bulabilirsiniz.
    
                                ### 2. **İki Örneklem Oran Testi (Proportions Z-Test)**
                                - **Amaç**: İki örneklem oran testi, iki grup arasındaki **oran farkını** test eder. Yani, iki grubun belirli bir olayı yaşama oranları arasında anlamlı bir fark olup olmadığını kontrol eder.
                                - **Varsayımlar**: 
                                  - Her iki grubun örnek büyüklüğü yeterince büyük olmalıdır.
                                  - Bağımsız örneklem.
    
                                Bu test, genellikle **kategorik** veriler için kullanılır. Eğer bir olayın iki grup arasındaki olasılığı ya da oranı arasında fark olup olmadığını test etmek isterseniz, bu testi kullanırsınız.
    
                                ### 3. **İki Testin Farklı Sonuçlar Vermesi**
                                İki test farklı şeyler ölçtüğünden, farklı sonuçlar vermesi mümkündür. Bunun nedeni şunlar olabilir:
    
                                1. **Veri Türleri ve Analiz Yaklaşımı Farklıdır**: İki örneklem T testi **kesintisiz (sayısal)** verilerle çalışırken, iki örneklem oran testi **kategorik** verilerle çalışır. Dolayısıyla, bir testin fark bulduğu bir durumda diğer test fark bulamayabilir. Örneğin:
                                   - İki grup arasındaki ortalama değerler anlamlı derecede farklı olabilir (T testi anlamlı sonuç verebilir).
                                   - Ancak, iki grup arasında olayın gerçekleşme oranları arasında anlamlı bir fark olmayabilir (oran testi anlamlı sonuç vermeyebilir).
    
                                2. **Veri Dağılımı**: İki grup arasındaki dağılımlar farklı olabilir. Örneğin, iki grubun ortalama değeri önemli ölçüde farklı olabilir ancak oranlar (örneğin, bir olayın gerçekleşme oranı) bu kadar fark yaratmayabilir.
    
                                3. **Örneklem Büyüklüğü**: Oran testinde örneklem büyüklüğü veya gözlenen olay sayısı yeterli büyüklükte olmayabilir. T testi daha duyarlı olabilirken, oran testi daha büyük örnek büyüklüklerine ihtiyaç duyabilir.
    
                                ### 4. **Yorumlama**
                                - **İki Örneklem T Testi Anlamlı, Oran Testi Anlamsız**: Bu durumda, iki grubun **ortalama** değerleri arasında fark olduğunu, ancak bu farkın oranlar açısından yeterince güçlü olmadığını söyleyebilirsiniz.
                                   - Örneğin, grupların ortalama yaşları arasında fark olabilir, ancak iki grubun belirli bir hastalığa yakalanma oranı açısından anlamlı bir fark yoktur.
                                   - Bir diğer olasılık ise, iki grup arasında olayın frekansı, oranlara göre farklılık göstermiyor olabilir ama sayısal ölçüm açısından gruplar önemli bir fark sergiliyor olabilir.
    
                                - **Nasıl Yorumlanmalı**: Bu durumda, sayısal bir değişkenin iki grup arasında ortalama bazında farklı olduğunu, ancak bu farkın olayların oranları (kategorik sonuçlar) üzerinde belirgin bir etkisi olmadığını not etmek önemlidir. Yani, ortalamalar açısından iki grup farklı olabilir, fakat oranlar açısından farklılık bulunmamaktadır.
    
                                ### Özet:
                                - **İki Örneklem T Testi**: İki grup arasındaki ortalamaların farkını test eder. Eğer anlamlıysa, gruplar arasında ortalama değerler arasında anlamlı bir fark vardır.
                                - **İki Örneklem Oran Testi**: İki grup arasındaki oran farkını test eder. Eğer anlamlı değilse, bu iki grubun belirli bir olaya yakalanma oranları arasında fark olmadığı anlamına gelir.
                                - **Farklı sonuçlar**: Bu durum, sayısal değerler (ortalama farkları) ile kategorik olayların gerçekleşme oranları arasındaki farkların farklı sonuçlar doğurabileceğini gösterir.
    
                                Bu nedenle, testlerin amacına uygun şekilde sonuçları yorumlayarak, verinizin doğasına göre hangi farkların daha önemli olduğunu değerlendirmelisiniz.""",
                                                  key=f"lebelebeleb_{mpcounter}")
                                cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, num_but_cat_arg,
                                                                                                  cat_but_car_arg)

                                all_cols = num_cols + cat_cols

                                # Bağımsız ve bağımlı değişken seçimi
                                ind_var = st.selectbox('Bağımsız değişkeni seçin (kategorik olmalı):',
                                                           options=cat_cols,
                                                           key=f"ind_var_{mpcounter}")
                                dep_var = st.selectbox('Bağımlı değişkeni seçin (nümerik olmalı):',
                                                           options=all_cols,
                                                           key=f"dep_var_{mpcounter}")

                                # Eğer herhangi bir seçim yapmazsa döngü sonlandırılabilir
                                if not ind_var:
                                    st.write("Herhangi bir değişken seçilmedi. Döngü sonlandırılıyor.")
                                    break

                                # Eğer herhangi bir seçim yapmazsa döngü sonlandırılabilir
                                if not dep_var:
                                    st.write("Herhangi bir değişken seçilmedi. Döngü sonlandırılıyor.")
                                    break

                                # Bağımsız değişkenin unique değerlerinin kontrolü
                                unique_values = df[ind_var].nunique()
                                if unique_values > 2:
                                    st.error(
                                            f"Bağımsız değişkenin ({ind_var}) unique değer sayısı {unique_values}. ANOVA testi yapılmalıdır!")
                                    break

                                # Normallik Varsayımı
                                st.write("1. Normallik Varsayımı")
                                p_values = []  # Normallik testinin p-value'larını saklamak için
                                for unique_value in df[ind_var].unique():
                                    test_stat, pvalue = shapiro(df.loc[df[ind_var] == unique_value, dep_var])
                                    st.write(f'Bağımsız değişken: {ind_var} - Unique Değer: {unique_value}')
                                    st.write(f'Test Stat = {test_stat:.4f}, p-value = {pvalue:.4f}')
                                    p_values.append(pvalue)  # Her bir grubun p-value değeri listeye eklenir

                                # Normallik varsayımı kontrolü
                                if all(p < 0.05 for p in p_values):  # Her iki p-value değeri de 0.05'ten küçükse
                                    st.write("H0 RED: Normallik varsayımı sağlanmamaktadır.")
                                    normallik_saglandi = False

                                else:
                                    st.write("H0 REDDEDİLEMEZ: Normallik varsayımı sağlanmaktadır.")
                                    normallik_saglandi = True

                                # Varyans Homojenliği
                                if normallik_saglandi:
                                    st.write("2. Varyans Homojenliği")
                                    test_stat, pvalue = levene(
                                            df.loc[df[ind_var] == df[ind_var].unique()[0], dep_var],
                                            df.loc[df[ind_var] == df[ind_var].unique()[1], dep_var]
                                        )
                                    st.write(
                                            f'Varyans Homojenliği Testi - Test Stat = {test_stat:.4f}, p-value = {pvalue:.4f}')
                                    if pvalue < 0.05:
                                        st.write("H0 RED: Varyans homojenliği sağlanmamaktadır.")
                                        equal_var = False
                                    else:
                                        st.write("H0 REDDEDİLEMEZ: Varyans homojenliği sağlanmaktadır.")
                                        equal_var = True
                                else:
                                    equal_var = False



                                # Ortalama ve Medyanların Gösterilmesi
                                st.write("Grupların Ortalama ve Medyanları:")
                                st.write(df.groupby(ind_var).agg({dep_var: ["mean", "median"]}))


                                # Testlerin uygulanması
                                st.write("Test Sonuçları:")
                                if normallik_saglandi:  # Normallik sağlanıyorsa T testi yapılacak
                                    test_stat, pvalue = ttest_ind(
                                            df.loc[df[ind_var] == df[ind_var].unique()[0], dep_var],
                                            df.loc[df[ind_var] == df[ind_var].unique()[1], dep_var],
                                            equal_var=equal_var
                                        )
                                    st.write(f'T Testi - Test Stat = {test_stat:.4f}, p-value = {pvalue:.4f}')
                                else:  # Normallik sağlanmıyorsa Mann-Whitney U testi kullanılacak
                                    test_stat, pvalue = mannwhitneyu(
                                            df.loc[df[ind_var] == df[ind_var].unique()[0], dep_var],
                                            df.loc[df[ind_var] == df[ind_var].unique()[1], dep_var]
                                        )
                                    st.write(
                                            f'Mann-Whitney U Testi - Test Stat = {test_stat:.4f}, p-value = {pvalue:.4f}')

                                # Sonuçların yorumlanması
                                if pvalue < 0.05:
                                    st.write(
                                            f"H0 RED Edildi: Seçilen bağımsız değişken ({ind_var}) ile bağımlı değişken ({dep_var}) arasında anlamlı fark vardır.")
                                else:
                                    st.write(
                                            f"H0 REDDEDİLEMEZ: Seçilen bağımsız değişken ({ind_var}) ile bağımlı değişken ({dep_var}) arasında anlamlı fark yoktur.")



                            ################################################################################################
                            ##################   İki Örneklem Oran Testi
                            ################################################################################################

                            # Eğer "İki Örneklem Oran Testi" seçilirse
                            if mp_category_choice == "İki Örneklem Oran Testi":

                                st.button("Yorumlama hk.", help="""Soru: iki örneklem T testi sonucunda anlamlı ilişki bulurken iki örneklem oran testinde anlamlı ilişki yoktur sonucuna ulaşıyorum. bu mümkün müdür? nasıl yorumlamam gerekiyor? Cevap: Evet, iki örneklem T testi ve iki örneklem oran testi farklı sonuçlar verebilir, ve bu durumu anlamak ve yorumlamak önemlidir. Bunun mümkün olup olmadığını ve nasıl yorumlanacağını daha ayrıntılı olarak açıklayayım:

### 1. **İki Örneklem T Testi**
- **Amaç**: İki örneklem T testi, iki grup arasındaki **ortalama farkını** test eder. Yani, iki grubun ortalamaları arasında anlamlı bir fark olup olmadığını kontrol eder.
- **Varsayımlar**: 
  - Verilerin normal dağılıma sahip olması.
  - Varyansların homojen olması (eşit varyans).
  - Bağımsız örneklem.
  
T testi, sayısal (kesintisiz) bir değişkenin iki grup arasında ortalamasını karşılaştırmak için kullanılır. Eğer gruplar arasında ortalama farkı önemliyse, bu test sonucunda anlamlı bir ilişki bulabilirsiniz.

### 2. **İki Örneklem Oran Testi (Proportions Z-Test)**
- **Amaç**: İki örneklem oran testi, iki grup arasındaki **oran farkını** test eder. Yani, iki grubun belirli bir olayı yaşama oranları arasında anlamlı bir fark olup olmadığını kontrol eder.
- **Varsayımlar**: 
  - Her iki grubun örnek büyüklüğü yeterince büyük olmalıdır.
  - Bağımsız örneklem.

Bu test, genellikle **kategorik** veriler için kullanılır. Eğer bir olayın iki grup arasındaki olasılığı ya da oranı arasında fark olup olmadığını test etmek isterseniz, bu testi kullanırsınız.

### 3. **İki Testin Farklı Sonuçlar Vermesi**
İki test farklı şeyler ölçtüğünden, farklı sonuçlar vermesi mümkündür. Bunun nedeni şunlar olabilir:

1. **Veri Türleri ve Analiz Yaklaşımı Farklıdır**: İki örneklem T testi **kesintisiz (sayısal)** verilerle çalışırken, iki örneklem oran testi **kategorik** verilerle çalışır. Dolayısıyla, bir testin fark bulduğu bir durumda diğer test fark bulamayabilir. Örneğin:
   - İki grup arasındaki ortalama değerler anlamlı derecede farklı olabilir (T testi anlamlı sonuç verebilir).
   - Ancak, iki grup arasında olayın gerçekleşme oranları arasında anlamlı bir fark olmayabilir (oran testi anlamlı sonuç vermeyebilir).

2. **Veri Dağılımı**: İki grup arasındaki dağılımlar farklı olabilir. Örneğin, iki grubun ortalama değeri önemli ölçüde farklı olabilir ancak oranlar (örneğin, bir olayın gerçekleşme oranı) bu kadar fark yaratmayabilir.

3. **Örneklem Büyüklüğü**: Oran testinde örneklem büyüklüğü veya gözlenen olay sayısı yeterli büyüklükte olmayabilir. T testi daha duyarlı olabilirken, oran testi daha büyük örnek büyüklüklerine ihtiyaç duyabilir.

### 4. **Yorumlama**
- **İki Örneklem T Testi Anlamlı, Oran Testi Anlamsız**: Bu durumda, iki grubun **ortalama** değerleri arasında fark olduğunu, ancak bu farkın oranlar açısından yeterince güçlü olmadığını söyleyebilirsiniz.
   - Örneğin, grupların ortalama yaşları arasında fark olabilir, ancak iki grubun belirli bir hastalığa yakalanma oranı açısından anlamlı bir fark yoktur.
   - Bir diğer olasılık ise, iki grup arasında olayın frekansı, oranlara göre farklılık göstermiyor olabilir ama sayısal ölçüm açısından gruplar önemli bir fark sergiliyor olabilir.
   
- **Nasıl Yorumlanmalı**: Bu durumda, sayısal bir değişkenin iki grup arasında ortalama bazında farklı olduğunu, ancak bu farkın olayların oranları (kategorik sonuçlar) üzerinde belirgin bir etkisi olmadığını not etmek önemlidir. Yani, ortalamalar açısından iki grup farklı olabilir, fakat oranlar açısından farklılık bulunmamaktadır.

### Özet:
- **İki Örneklem T Testi**: İki grup arasındaki ortalamaların farkını test eder. Eğer anlamlıysa, gruplar arasında ortalama değerler arasında anlamlı bir fark vardır.
- **İki Örneklem Oran Testi**: İki grup arasındaki oran farkını test eder. Eğer anlamlı değilse, bu iki grubun belirli bir olaya yakalanma oranları arasında fark olmadığı anlamına gelir.
- **Farklı sonuçlar**: Bu durum, sayısal değerler (ortalama farkları) ile kategorik olayların gerçekleşme oranları arasındaki farkların farklı sonuçlar doğurabileceğini gösterir.

Bu nedenle, testlerin amacına uygun şekilde sonuçları yorumlayarak, verinizin doğasına göre hangi farkların daha önemli olduğunu değerlendirmelisiniz.""", key=f"labalabalab_{mpcounter}")

                                # İşlem yapmak istiyorsa, kullanıcıdan bağımsız değişken seçmesini isteyelim
                                oran_selected_column_meas = st.selectbox(
                                    'Bağımsız değişkeni seçin:',
                                    options=df.columns,
                                    key=f"oran_selected_column_meas_{mpcounter}"  # mp_counter ile benzersiz key
                                )

                                # Eğer herhangi bir seçim yapmazsa döngü sonlandırılabilir
                                if not oran_selected_column_meas:
                                    st.write("Herhangi bir değişken seçilmedi. Döngü sonlandırılıyor.")
                                    break

                                # Yeni eklenen değişkenin unique değer sayısını kontrol edelim
                                oran_unique_values_meas = df[oran_selected_column_meas].unique()
                                oran_num_unique_values_meas = len(oran_unique_values_meas)

                                if oran_num_unique_values_meas == 1:
                                    # Tek unique değer varsa
                                    st.write(f"Tek unique değer bulunmaktadır: {oran_unique_values_meas[0]}")

                                elif oran_num_unique_values_meas == 2:
                                    # İki unique değer varsa proportions_ztest testi yapılacak
                                    st.write(f"İki unique değer bulunmaktadır. Değerler: {oran_unique_values_meas}")

                                    # Hedef değişken için kullanıcıdan seçim alalım
                                    oran_fe_target_var_meas = st.selectbox(
                                        'Hedef değişkeni seçin:',
                                        options=df.columns,
                                        key=f"oran_fe_target_var_meas_{mpcounter}"  # mp_counter ile benzersiz key
                                    )

                                    # İki grup için başarı oranlarını hesaplayalım
                                    group1 = oran_unique_values_meas[0]
                                    group2 = oran_unique_values_meas[1]

                                    # Grup 1 ve Grup 2'nin başarı oranları ve gözlem sayıları
                                    group1_success = df.loc[
                                        df[oran_selected_column_meas] == group1, oran_fe_target_var_meas].sum()
                                    group2_success = df.loc[
                                        df[oran_selected_column_meas] == group2, oran_fe_target_var_meas].sum()

                                    group1_nobs = \
                                    df.loc[df[oran_selected_column_meas] == group1, oran_fe_target_var_meas].shape[0]
                                    group2_nobs = \
                                    df.loc[df[oran_selected_column_meas] == group2, oran_fe_target_var_meas].shape[0]

                                    # Başarı oranı hesaplama
                                    group1_success_rate = group1_success / group1_nobs if group1_nobs != 0 else 0
                                    group2_success_rate = group2_success / group2_nobs if group2_nobs != 0 else 0

                                    # Sonuçları tablo şeklinde göstermek için DataFrame oluşturalım
                                    success_df = pd.DataFrame({
                                        "Grup": [group1, group2],
                                        "Başarı Sayısı": [group1_success, group2_success],
                                        "Gözlem Sayısı": [group1_nobs, group2_nobs],
                                        "Başarı Oranı": [group1_success_rate, group2_success_rate]
                                    })

                                    st.write("Grup bazında başarı oranları:")
                                    st.dataframe(success_df)

                                    # İstatistiki analiz: Oranlar testi (proportions_ztest) tüm unique değerler için yapılacak
                                    for value in oran_unique_values_meas:
                                        oran_count_1 = df.loc[
                                            df[oran_selected_column_meas] == value, oran_fe_target_var_meas].sum()
                                        oran_count_0 = df.loc[
                                            df[oran_selected_column_meas] != value, oran_fe_target_var_meas].sum()
                                        oran_nobs_1 = \
                                        df.loc[df[oran_selected_column_meas] == value, oran_fe_target_var_meas].shape[0]
                                        oran_nobs_0 = \
                                        df.loc[df[oran_selected_column_meas] != value, oran_fe_target_var_meas].shape[0]

                                        if oran_nobs_0 == 0 or oran_nobs_1 == 0:  # Boş kümelerde test yapmamak için
                                            st.write(f"Yeterli veri yok: {value} için test yapılmadı.")
                                            continue

                                        oran_test_stat, oran_pvalue = proportions_ztest(
                                            count=[oran_count_1, oran_count_0],
                                            nobs=[oran_nobs_1, oran_nobs_0])

                                    st.write(f'H0 için Test Stat = {oran_test_stat:.4f}, p-value = {oran_pvalue:.4f}')
                                    if oran_pvalue < 0.05:
                                        st.write(
                                            f"H0 hipotezi reddedildi: Seçilen bağımsız değişken ({oran_selected_column_meas}) ile seçilen hedef değişken ({oran_fe_target_var_meas}) arasında fark vardır.")
                                    else:
                                        st.write(
                                            f"H0 hipotezi reddedilemedi: Seçilen bağımsız değişken ({oran_selected_column_meas}) ile seçilen hedef değişken ({oran_fe_target_var_meas}) arasında fark yoktur.")

                                elif oran_num_unique_values_meas > 2:
                                    # İkiden fazla unique değer varsa ANOVA testi yapılacaktır yazısı gösterilecek
                                    st.write(
                                        f"{oran_num_unique_values_meas} unique değer bulunmaktadır. ANOVA testi yapınız.")

                            ################################################################################################
                            ##################   ANOVA
                            ################################################################################################

                            if mp_category_choice == "ANOVA":

                                cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, num_but_cat_arg,
                                                                                              cat_but_car_arg)

                                all_cols = num_cols + cat_cols

                                # Bağımsız ve bağımlı değişken seçimi
                                anova_ind_var = st.selectbox('Bağımsız değişkeni seçin (kategorik olmalı):',
                                                             options=cat_cols, key=f"anova_ind_var_{mpcounter}")
                                anova_dep_var = st.selectbox('Bağımlı değişkeni seçin (nümerik olmalı):',
                                                             options=all_cols, key=f"anova_dep_var_{mpcounter}")

                                # NaN değerleri ayrı bir grup olarak inceleyelim
                                unique_groups = df[anova_ind_var].unique().tolist()

                                # NaN değerleri manuel olarak ekliyoruz
                                if df[anova_ind_var].isna().sum() > 0:
                                    unique_groups.append(np.nan)  # NaN'ı gruba ekle

                                group_sizes = {
                                    group: len(df.loc[df[anova_ind_var] == group, anova_dep_var]) if pd.notna(group)
                                    else len(df.loc[df[anova_ind_var].isna(), anova_dep_var]) for group in
                                    unique_groups}

                                # Her grup için boyutları yazdırıyoruz
                                for group, size in group_sizes.items():
                                    if pd.isna(group):
                                        st.write(f"Group: NaN, Size: {size}")
                                    else:
                                        st.write(f"Group: {group}, Size: {size}")

                                # Eğer herhangi bir grup 3'ten küçükse, uyarı verip işlemi durduruyoruz
                                if any(size < 3 for size in group_sizes.values()):
                                    st.write("Her bir grupta en az 3 veri bulunmalıdır. Test işlemi yapılamıyor.")
                                    st.stop()  # Kod burada duracak ve devam etmeyecek

                                else:
                                    # Normallik Varsayımı (Shapiro Testi)
                                    st.write("1. Normallik Varsayımı")
                                    normallik_kontrolu = []
                                    for group in unique_groups:
                                        anovanom_pvalue = shapiro(df.loc[df[anova_ind_var] == group, anova_dep_var])[1]
                                        st.write(f'Grup: {group} - p-value: {anovanom_pvalue:.4f}')
                                        normallik_kontrolu.append(anovanom_pvalue)
                                        if anovanom_pvalue < 0.05:
                                            st.write(f"Grup {group} için normallik varsayımı sağlanmamaktadır.")
                                        else:
                                            st.write(f"Grup {group} için normallik varsayımı sağlanmaktadır.")

                                    # Varyans Homojenliği (Levene Testi)
                                    st.write("2. Varyans Homojenliği")
                                    test_stat, anovavar_pvalue = levene(
                                        *[df.loc[df[anova_ind_var] == group, anova_dep_var] for group in unique_groups]
                                    )
                                    st.write(
                                        f'Varyans Homojenliği Testi - Test Stat = {test_stat:.4f}, p-value = {anovavar_pvalue:.4f}')
                                    if anovavar_pvalue < 0.05:
                                        st.write("Varyans homojenliği varsayımı sağlanmamaktadır.")
                                    else:
                                        st.write("Varyans homojenliği varsayımı sağlanmaktadır.")

                                    # Ortalama ve Medyanların Gösterilmesi
                                    st.write("3. Grupların Ortalama ve Medyanları:")
                                    st.write(df.groupby(anova_ind_var).agg({anova_dep_var: ["mean", "median"]}))

                                    # Varsayımlara göre testlerin seçilmesi
                                    st.write("4. Uygulanacak Test")
                                    if any(p < 0.05 for p in normallik_kontrolu) or anovavar_pvalue < 0.05:
                                        st.write(
                                            "Normallik veya varyans homojenliği sağlanmadığından Non-Parametrik ANOVA testi uygulanacak.")

                                        # Kruskal-Wallis testi
                                        test_stat, anovanonp_pvalue = kruskal(
                                            *[df.loc[df[anova_ind_var] == group, anova_dep_var] for group in
                                              unique_groups]
                                        )
                                        st.write(
                                            f'Nonparametrik ANOVA Testi (Kruskal) - Test Stat = {test_stat:.4f}, p-value = {anovanonp_pvalue:.4f}')
                                        if anovanonp_pvalue < 0.05:
                                            st.write(
                                                f"H0 reddedildi: Bağımsız değişken ({anova_ind_var}) ile bağımlı değişken ({anova_dep_var}) arasında anlamlı fark vardır.")
                                        else:
                                            st.write(
                                                f"H0 reddedilemedi: Bağımsız değişken ({anova_ind_var}) ile bağımlı değişken ({anova_dep_var}) arasında anlamlı fark yoktur.")
                                    else:
                                        st.write(
                                            "Normallik ve varyans homojenliği sağlandığından Parametrik ANOVA testi (f-oneway) uygulanacak.")
                                        test_stat, anovap_pvalue = f_oneway(
                                            *[df.loc[df[anova_ind_var] == group, anova_dep_var] for group in
                                              unique_groups]
                                        )
                                        st.write(
                                            f'Parametrik ANOVA Testi - Test Stat = {test_stat:.4f}, p-value = {anovap_pvalue:.4f}')
                                        if anovap_pvalue < 0.05:
                                            st.write(
                                                f"H0 reddedildi: Bağımsız değişken ({anova_ind_var}) ile bağımlı değişken ({anova_dep_var}) arasında anlamlı fark vardır.")
                                        else:
                                            st.write(
                                                f"H0 reddedilemedi: Bağımsız değişken ({anova_ind_var}) ile bağımlı değişken ({anova_dep_var}) arasında anlamlı fark yoktur.")

                                    # Tukey HSD Testi
                                    st.write(
                                        "Anlamlı fark bulundu. Hangi gruptan kaynaklandığını anlamak için Tukey HSD testi uygulanacak.")
                                    tk_value = st.number_input(
                                        'Tukey HSD Testi için tk (yanlış pozitiflik riski) değerini girin (genel kabul değer 0.05 - default)',
                                        value=0.05, key=f"tk_value_{mpcounter}")

                                    # Tukey HSD testi
                                    comparison = MultiComparison(df[anova_dep_var], df[anova_ind_var])
                                    tukey = comparison.tukeyhsd(tk_value)

                                    # Tukey sonuçlarını DataFrame olarak gösterelim
                                    tukey_df = pd.DataFrame(data=tukey.summary().data[1:],
                                                            columns=tukey.summary().data[0])
                                    st.write(tukey_df)




                            # Kullanıcıya başka bir işlem yapmak isteyip istemediği soruluyor
                            continue_or_not = st.radio("Başka bir işlem yapmak istiyor musunuz?", ("Evet", "Hayır"),
                                                       index=1,
                                                       key=f"mpradio_{mpcounter}")  # mp_counter ile benzersiz key



                            if continue_or_not == "Hayır":
                                st.write("Döngü sonlandırıldı.")
                                break

                        mpcounter += 1





                    ####################################
                    ################ PRE-OVERVIEW ÜST TABI - Analysis bölümü - Correlation Matrix alt tabı
                    ####################################

                    with vtab5.expander("Correlation Analysis"):  # Correlation Matrix

                            corr_anal = st.checkbox("Perform Correlation?")



                            if corr_anal:

                                remove_vars = st.radio(
                                    "Do you want to remove any variables before creating the correlation matrix?",
                                    ("No", "Yes"))

                                if remove_vars == "Yes":
                                    # Allow user to select variables to remove
                                    variables_to_remove = st.multiselect(
                                        "Select variables to remove before creating the initial correlation matrix:",
                                        df.columns)
                                    # Create a local copy of the DataFrame and drop selected variables
                                    local_df = df.drop(columns=variables_to_remove)
                                else:
                                    # If no variables are removed, use the original DataFrame
                                    variables_to_remove = []
                                    local_df = df.copy()

                                # Filter the DataFrame to include only numeric columns
                                numeric_df = local_df.select_dtypes(include=[np.number])

                                # Calculate the correlation matrix
                                corr = numeric_df.corr()
                                cor_matrix = corr.abs()


                                # Ask the user if they want to plot the correlation heatmap
                                plot_corr = st.checkbox("Plot Correlation Heatmap")

                                # Optionally plot the correlation matrix
                                if plot_corr:
                                    fig, ax = plt.subplots(figsize=(15, 15))
                                    sns.heatmap(corr, cmap="RdBu", ax=ax)
                                    st.pyplot(fig)

                                # Display the correlation matrix and list of highly correlated columns
                                st.write("Correlation Matrix:", cor_matrix)


                                ######## Allow the user to select two numerical variables to plot the correlation scatter
                                #
                                binary_correlation_scatter = st.checkbox("Do you want to see two variables correlation scatter?")

                                if binary_correlation_scatter:
                                    st.write("## Correlation Scatter Plot")
                                    numerical_columns = numeric_df.select_dtypes(include=[np.number]).columns
                                    selected_columns = st.multiselect(
                                        "Select two numerical variables to see their correlation scatter plot:",
                                        numerical_columns, default=numerical_columns[:2])

                                    # Ensure that exactly two columns are selected
                                    if len(selected_columns) == 2:
                                        # Scatter plot of the two selected columns
                                        st.write(f"Scatter plot of {selected_columns[0]} vs {selected_columns[1]}")
                                        fig, ax = plt.subplots()
                                        numeric_df.plot.scatter(x=selected_columns[0], y=selected_columns[1], ax=ax)
                                        st.pyplot(fig)

                                        # Display the correlation value
                                        corr_value = numeric_df[selected_columns[0]].corr(numeric_df[selected_columns[1]])
                                        st.write(
                                            f"Correlation between {selected_columns[0]} and {selected_columns[1]}: {corr_value:.2f}")

                                        if corr_value > 0:
                                            st.write(
                                                f"As {selected_columns[0]} increases, {selected_columns[1]} tends to increase.")
                                        else:
                                            st.write(
                                                f"As {selected_columns[0]} increases, {selected_columns[1]} tends to decrease.")
                                    else:
                                        st.write("Please select exactly two numerical variables.")




                                ######## highly_correlated_columns
                                #
                                highly_correlated_columns = st.checkbox("Do you want to see highly correlated columns?")

                                if highly_correlated_columns:
                                    # Kullanıcıdan corr_th için input al
                                    corr_th = st.number_input("Enter correlation threshold (corr_th):", min_value=0.0,
                                                              max_value=1.0, value=0.9, step=0.000001)

                                    if highly_correlated_columns:
                                        # Identify highly correlated columns
                                        upper_triangle_matrix = cor_matrix.where(
                                            np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
                                        drop_list = [col for col in upper_triangle_matrix.columns if
                                                     any(upper_triangle_matrix[col] > corr_th)]

                                        # If there are columns to drop, show the correlation matrix after dropping them
                                        if drop_list:
                                            st.write(f"Highly correlated columns (corr_th > {corr_th}):", drop_list)

                                            all_vars_to_drop = list(set(variables_to_remove + drop_list))

                                            # Kullanıcıya drop_list içindeki değişkenlerden seçim yapmasına izin ver
                                            columns_to_drop = st.multiselect(
                                                "Select columns to drop from the highly correlated list and the list if you have selected the variables to remove before creating the initial correlation matrix:", all_vars_to_drop)

                                            if columns_to_drop:
                                                # Create a local copy of the DataFrame and drop the combined list of columns
                                                df = df.drop(columns=columns_to_drop)

                                                st.write("Current Dataframe:")
                                                st.write(df)







                ##################################################################3
                ################# FEATURE ENGINEERING ÜST TABI
                ##################################################################3

                p2_df = df.copy()

                # Encoding listesi
                enc_feature_options = [
                    "Hiçbir işlem yapmak istemiyorum",  # Varsayılan seçenek
                    "Label Encoding",
                    "One-Hot Encoding",
                    "Rare Encoding"
                ]

                # Scaling listesi
                scaling_feature_options = [
                    "Hiçbir işlem yapmak istemiyorum",  # Varsayılan seçenek
                    "StandardScaler",
                    "RobustScaler",
                    "MinMaxScaler"
                ]

                # Outliers listesi
                outliers_feature_options = [
                    "Hiçbir işlem yapmak istemiyorum",  # Varsayılan seçenek
                    "Outliers Detections & Handling",
                    "Local Outlier Factor & Handling"
                ]

                # Missing listesi
                missing_feature_options = [
                    "Hiçbir işlem yapmak istemiyorum",  # Varsayılan seçenek
                    "Missing Value Detection & Analysis",
                    "Eksik verinin rassallığı",
                    "Handling - Removing & Imputations"
                ]

                # MISSING VALUES - Handling - Basic Imputations listesi
                missing_bi_feature_options = [
                    "Hiçbir işlem yapmak istemiyorum",  # Varsayılan seçenek
                    "Removing the Value(s)",
                    "Numeric - Impute with Mean() Value",
                    "Numeric - Impute with Mode() Value",
                    "Numeric - Impute with Median() Value",
                    "Numeric - Impute with Custom Value",
                    "Numeric - Impute with Random Values Between 2 Bounds",
                    "Numeric - Impute with Interpolation",
                    "Numeric - Impute with group based mean()/median()/mode() Value",
                    "Categoric - Impute with Mode() Value",
                    "Categoric - Impute with Custom String Value",
                    "Categoric - Impute with Random String Value",
                    "kNN Based Imputations"
                ]


                # Structure listesi
                structure_feature_options = [
                    "Hiçbir işlem yapmak istemiyorum",  # Varsayılan seçenek
                    "Quick Remove Variable(s)",
                    "Time Variable Processing"
                ]


                # Feature extraction listesi
                feature_options = [
                    "Hiçbir işlem yapmak istemiyorum",  # Varsayılan seçenek
                    "Time değişkeni türetmek",
                    "NaN bool değişkeni türetmek",
                    "Koşullu matematiksel operasyonlar ile değişken türetmek",
                    "Nümerik değişkeni bilinen sınırlara bölerek değişken türetmek",
                    "Nümerik değişkeni çeyreklik sınırlara bölerek değişken türetmek",
                    "Nümerik değişkenlerde özellik etkileşimiyle değişken türetmek",
                    "Kategorik değişkenlerde harf sayısı ile değişken türetmek",
                    "Kategorik değişkenlerde kelime sayısı ile değişken türetmek",
                    "Belirli bir string ifade içerenlerle değişken türetmek",
                    "Regex ile metin analizi ile değişken türetmek"
                ]

                selected_operations = []

                processing_tab.subheader("FEATURE ENGINEERING & DATA PROCESSING OPTIONS")

                counter = 1
                while True:
                    with processing_tab.expander(f"İşlem Seçimi {counter}", expanded=True):
                        fecol1, fecol2 = st.columns(2)

                        # Kullanıcıya önce kategori seçimi sunuluyor
                        category_choice = fecol1.selectbox(
                            "Lütfen işlem kategorisini seçin:",
                            ["Hiçbir işlem yapmak istemiyorum", "Structure", "Missing Values", "Outliers", "Encoding", "Feature Extraction", "Scaling"],
                            key=f"selectbox_category_{counter}"
                        )


                        ################################################################################################
                        ##################   MISSING VALUES ÜST BAŞLIĞI
                        ################################################################################################

                        # Eğer Missing Values seçilirse encoding seçenekleri sunuluyor
                        if category_choice == "Missing Values":
                            missing_selected_feature = fecol1.selectbox(
                                "Missing values işlemi seçin:",
                                options=missing_feature_options,
                                key=f"missing_selectbox_{counter}_feature"
                            )


                            ##############################
                            #### MISSING VALUES - Missing Value Detection & Analysis
                            ##############################
                            # Quick Remove Variable işlemi seçilirse
                            if missing_selected_feature == "Missing Value Detection & Analysis":
                                ##############################
                                ## Missing Values Analysis Section
                                ##############################

                                fecol1.write("What Literature Says?")
                                fecol1.markdown("""
                                <style>
                                    .highlight {
                                        background-color: #f0f0f0; /* Light gray background */
                                        padding: 10px;
                                        border-radius: 5px;
                                        color: #333; /* Dark gray text color */
                                    }
                                    .quote {
                                        background-color: #eaf1f8; /* Light blue background */
                                        padding: 10px;
                                        border-left: 5px solid #007bff; /* Blue left border */
                                        color: #333; /* Dark gray text color */
                                        margin: 10px 0;
                                    }
                                    .note {
                                        background-color: #fff3cd; /* Yellow background */
                                        padding: 10px;
                                        border-left: 5px solid #ffc107; /* Yellow left border */
                                        color: #333; /* Dark gray text color */
                                        margin: 10px 0;
                                    }
                                </style>

                                <div class="highlight">
                                    <h3>Missing Value Detection</h3>
                                    <p><b>"The idea of imputation is both seductive and dangerous"</b> - R.J.A Little & D.B. Rubin.</p>
                                    <p><b>Directly removing observations with missing values and analyzing randomness can reduce the reliability of statistical inferences and modeling efforts.</b> - Alpar, 2011. 
                                    <br><i>Uygulamalı Çok Değişkenli İstatistiksel Yöntemler, Reha Alpar's book</i></p>
                                    <p><b>For missing observations to be directly removed from the dataset, the missingness in the dataset is expected to be either partially or completely random. If it is caused by structural problems related to the variables, the removal may lead to serious biases.</b> - Tabachnick & Fidell, 1996.</p>
                                </div>

                                <div class="note">
                                    <b>Note:</b> Handling missing data and imputation methods are critical stages in data science. Proper management of this information can directly impact the reliability of your modeling results.
                                </div>
                                """, unsafe_allow_html=True)

                                # Kullanıcıya Missing Value Detection yapmak isteyip istemediğini soralım, varsayılan olarak "Hayır" seçili
                                perform_missing_value_detection = fecol1.radio(
                                    f"Missing Value Detection yapmak istiyor musunuz? {counter}",
                                    ("Hayır", "Evet"),
                                    index=0,
                                    key=f"perform_missing_value_detection_{counter}")

                                # Eğer kullanıcı "Evet" derse Missing Value Detection işlemini yapalım
                                if perform_missing_value_detection == "Evet":


                                    # Missing value detection and analysis
                                    fecol1.write(f"**Eksik Değerlerin Tespiti ve Analizi:**")

                                    has_missing = p2_df.isnull().values.any()
                                    total_missing = p2_df.isnull().sum().sum()
                                    fecol1.write(f"**Eksik gözlem var mı?** {has_missing}")
                                    fecol1.write(f"**Toplam eksik gözlem sayısı:** {total_missing}")

                                    # Missing and filled values per column
                                    missing_per_col = p2_df.isnull().sum().sort_values(ascending=False)
                                    filled_per_col = p2_df.notnull().sum().sort_values(ascending=False)


                                    femcol1, femcol2 = fecol1.columns(2)

                                    femcol1.write("**Her bir değişkendeki eksik değer sayısı:**")
                                    femcol1.write(missing_per_col.transpose())
                                    femcol2.write("**Her bir değişkendeki dolu değer sayısı:**")
                                    femcol2.write(filled_per_col.transpose())

                                    # Observations with at least one missing value
                                    femcol1.write(f"**En az bir eksik değere sahip gözlem birimleri:**")
                                    femcol1.write(p2_df[p2_df.isnull().any(axis=1)])

                                    # Complete observations
                                    femcol2.write(f"**Tam olan gözlem birimleri:**")
                                    femcol2.write(p2_df[p2_df.notnull().all(axis=1)])

                                    # Missing value ratios
                                    missing_ratios = (p2_df.isnull().sum() / p2_df.shape[0] * 100).sort_values(
                                        ascending=False)
                                    femcol1.write("**Eksik değerlerin oranı (%) :**")
                                    femcol1.write(missing_ratios)

                                    # Storing columns with missing values
                                    na_cols = [col for col in p2_df.columns if p2_df[col].isnull().sum() > 0]
                                    # na_cols listesini DataFrame'e dönüştürelim
                                    na_cols_df = pd.DataFrame(na_cols, columns=["Column"])

                                    # DataFrame'i fecol2'de yazdıralım
                                    femcol2.write("**Missing Value içeren sütunlar:**")
                                    femcol2.write(na_cols_df)


                                ##############################
                                ## Missing Value Detailed Analysis Section
                                ##############################

                                detayli_analiz1 = fecol2.radio(
                                    f"Eksik veri yapısını incelemek istiyor musunuz? {counter}:",
                                    ("Hayır", "Evet"),
                                    index=0,
                                    key=f"detayli_analiz1_{counter}")

                                if detayli_analiz1 == "Evet":
                                    fecol2.subheader("Eksik Veri Yapısının İncelenmesi")

                                    # Eksik veri inceleme fonksiyonu
                                    def missing_vs_target(dataframe, target, na_columns):
                                        temp_df = dataframe.copy()  # DataFrame'in bir kopyasını oluşturuyoruz.

                                        # NA değerlerine göre flag sütunları ekliyoruz
                                        for col in na_columns:
                                            temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

                                        na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

                                        results = []
                                        for col in na_flags:
                                            target_mean = temp_df.groupby(col)[target].mean()
                                            count = temp_df.groupby(col)[target].count()
                                            result = pd.DataFrame({"TARGET_MEAN": target_mean, "Count": count})
                                            results.append(result)

                                        return results

                                    # Eksik veri görselleştirme seçenekleri
                                    fecol2.write(f"**Eksik Veri Bar Grafiği:**")
                                    fig, ax = plt.subplots()
                                    msno.bar(p2_df, ax=ax)
                                    fecol2.pyplot(fig)

                                    fecol2.write(f"**Eksik Veri Matris Görsellemesi:**")
                                    fig, ax = plt.subplots()
                                    msno.matrix(p2_df, ax=ax)
                                    fecol2.pyplot(fig)

                                    fecol2.write(f"**Eksik Veri Korelasyon (Heatmap) Analizi:**")
                                    fig, ax = plt.subplots()
                                    msno.heatmap(p2_df, ax=ax)
                                    fecol2.pyplot(fig)

                                else:
                                    fecol2.write("İnceleme yapılmamıştır.")



                                detayli_analiz2 = fecol2.radio(
                                        f"Eksik değerlerin bağımlı değişken ile ilişkisini incelemek istiyor musunuz? {counter}:",
                                        ("Hayır", "Evet"),
                                        index=0,
                                        key=f"detayli_analiz2_{counter}")

                                if detayli_analiz2 == "Evet":
                                    fecol2.subheader("Eksik Veri Yapısının İncelenmesi")

                                    fecol2.subheader(
                                        f"**Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi:**")

                                    target_column = fecol2.selectbox(f"Bağımlı Değişkeni Seçin {counter}",
                                                                        p2_df.columns)


                                    # Eksik veri bulunan sütunları listeleme
                                    def missing_values_table(dataframe, na_name=False):
                                        na_columns = [col for col in dataframe.columns if
                                                        dataframe[col].isnull().sum() > 0]

                                        n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
                                        ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[
                                            0] * 100).sort_values(
                                            ascending=False)
                                        missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1,
                                                                   keys=['n_miss', 'ratio'])
                                        print(missing_df, end="\n")

                                        if na_name:
                                            return na_columns

                                    na_cols = missing_values_table(p2_df, True)
                                    if len(na_cols) > 0:
                                        fecol2.write(f"**Eksik veri içeren sütunlar:** {', '.join(na_cols)}")

                                        missing_analysis = missing_vs_target(p2_df, target_column, na_cols)

                                        fecol2.write(
                                                "**NA olanlar için 1 yazanlara bakacağım. 0 olanlar ise dolu olanlar:**")
                                        for result in missing_analysis:
                                                fecol2.write(result)
                                    else:
                                            fecol2.write(f"Veri setinde eksik veri bulunmamaktadır.")
                                else:
                                    fecol2.write("İnceleme yapılmamıştır.")

                            ##############################
                            #### MISSING VALUES - Eksik Verinin Rassallığı
                            ##############################
                            if missing_selected_feature == "Eksik verinin rassallığı":
                                fecol1.write("Eksik verinin rassallığını test edin")

                                # Kullanıcıya analiz türünü seçtiriyoruz
                                randomness_test_type = fecol1.selectbox(
                                    "Eksik verinin rastgele olup olmadığını hangi yöntemle incelemek istersiniz?",
                                    options=["MCAR Test (Little's MCAR)", "Missing Value Visualization"],
                                    key=f"randomness_test_type_{counter}"
                                )

                                ##############################
                                #### Little's MCAR Test (MCAR Testi)
                                ##############################
                                if randomness_test_type == "MCAR Test (Little's MCAR)":
                                    fecol1.write("Little's MCAR Testi yapılacak.")

                                    # Kullanıcıya veriyi seçtiriyoruz
                                    target_variable = fecol1.selectbox(
                                        "Hangi bağımlı değişken üzerinde analiz yapmak istersiniz?",
                                        options=p2_df.columns,
                                        key=f"mcar_test_target_{counter}"
                                    )

                                    # Little's MCAR testi için verinin hazırlanması ve uygulama
                                    fecol1.warning(
                                        "Bu aşama örnek amaçlıdır. MCAR testi için R dilini veya başka istatistik paketlerini kullanmalısınız.")
                                    fecol1.write(
                                        "Python'da doğrudan bir MCAR testi yoktur, ancak R paketlerinde uygulanabilir.")

                                ##############################
                                #### Missing Value Visualization
                                ##############################
                                elif randomness_test_type == "Missing Value Visualization":
                                    fecol1.write("Eksik verilerin görselleştirilmesi yapılacak.")

                                    # Görselleştirme seçeneklerini sunuyoruz
                                    visualization_type = fecol1.radio(
                                        "Hangi görselleştirme yöntemini kullanmak istersiniz?",
                                        options=["Matrix", "Heatmap", "Bar Plot"],
                                        key=f"missing_visualization_{counter}"
                                    )

                                    if visualization_type == "Matrix":
                                        fecol1.write(f"**Eksik Veri Matris Görsellemesi:**")
                                        fig, ax = plt.subplots(figsize=(8, 5))
                                        msno.matrix(p2_df, ax=ax)
                                        fecol1.pyplot(fig)

                                    elif visualization_type == "Heatmap":
                                        fecol1.write(f"**Eksik Veri Korelasyon (Heatmap) Görsellemesi:**")
                                        fig, ax = plt.subplots(figsize=(8, 5))
                                        msno.heatmap(p2_df, ax=ax)
                                        fecol1.pyplot(fig)

                                    elif visualization_type == "Bar Plot":
                                        fecol1.write(f"**Eksik Veri Bar Grafiği:**")
                                        fig, ax = plt.subplots(figsize=(8, 5))
                                        msno.bar(p2_df, ax=ax)
                                        fecol1.pyplot(fig)

                                ##############################
                                #### Missing Value Logistic Regression (Optional)
                                ##############################
                                use_logistic_regression = fecol1.checkbox(
                                    "Eksik veriler için Logistic Regression yapmak ister misiniz?",
                                    key=f"logistic_regression_{counter}"
                                )

                                if use_logistic_regression:
                                    fecol1.write("Eksik verilerin Logistic Regression ile incelenmesi yapılacak.")
                                    target_var = fecol1.selectbox("Bağımlı değişkeni seçin:", options=p2_df.columns,
                                                                  key=f"lr_target_var_{counter}")

                                    # Logistic Regression için uygun veri hazırlığı (sadece eksik olan değişkenler üzerinde)
                                    missing_cols = [col for col in p2_df.columns if p2_df[col].isnull().sum() > 0]

                                    if missing_cols:
                                        fecol1.write(f"**Eksik veri bulunan sütunlar:** {missing_cols}")

                                        # Logistic Regression modelini eksik değerlere uygulayacağız
                                        from sklearn.linear_model import LogisticRegression
                                        from sklearn.model_selection import train_test_split

                                        # Eksik veri sütunları üzerinden modelleme yapılacak
                                        for col in missing_cols:
                                            p2_df[f'{col}_missing'] = p2_df[col].isnull().astype(
                                                int)  # Eksik veri bayrağı
                                            X = p2_df.dropna(subset=[col]).drop(columns=[col, target_var])
                                            y = p2_df.dropna(subset=[col])[f'{col}_missing']

                                            # Veri bölme işlemi
                                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                                                                random_state=42)

                                            # Modelleme
                                            model = LogisticRegression()
                                            model.fit(X_train, y_train)
                                            score = model.score(X_test, y_test)

                                            fecol1.write(
                                                f"{col} sütunu için Logistic Regression başarı skoru: {score:.2f}")

                            ##############################
                            #### MISSING VALUES - Handling - Removing & Imputations
                            ##############################
                            # Quick Remove Variable işlemi seçilirse
                            if missing_selected_feature == "Handling - Removing & Imputations":

                                missing_bi_selected_feature = fecol1.selectbox(
                                    "Missing values işlemi seçin:",
                                    options=missing_bi_feature_options,
                                    key=f"missing_bi_selectbox_{counter}_feature"
                                )

                                p2_cat_cols, p2_num_cols, p2_cat_but_car, p2_num_but_cat = grab_col_names(p2_df)

                                # Eksik değerlerle başa çıkma seçeneği için benzersiz key eklenmiş
                                handle_missing = fecol1.radio(
                                    "Eksik değerlerle baş etmek istiyor musunuz?",
                                    ("Hayır", "Evet"),
                                    key=f"handle_missing_radio_{counter}"
                                )

                                if handle_missing == "Hayır":
                                    fecol1.write("İşlem yapılmadı.")
                                    fecol1.dataframe(p2_df)
                                else:

                                    ###########################################################
                                    #### MISSING VALUES - Handling - Removing & Imputations // Removing the Value(s)
                                    ###########################################################
                                    if missing_bi_selected_feature == "Removing the Value(s)":

                                        na_cols = [col for col in p2_df.columns if p2_df[col].isnull().sum() > 0]


                                        # Multiselect widget'ı için benzersiz key eklenmiş
                                        selected_remove = fecol1.multiselect(
                                            "Select variables for removal:",
                                            na_cols,
                                            key=f"selected_remove_multiselect_{counter}"
                                        )
                                        if selected_remove:
                                            # Eksik değerlerin sayısını saklamak için bir sözlük başlatalım
                                            missing_counts = {}

                                            # Seçilen her bir değişken için eksik değer sayısını hesaplayalım
                                            for col in selected_remove:
                                                missing_count = p2_df[col].isna().sum()
                                                missing_counts[col] = missing_count

                                            # Seçilen sütunlardaki eksik değerlere sahip satırları çıkaralım
                                            p2_df.dropna(subset=selected_remove, inplace=True)

                                            # Her sütun için silinen eksik değerlerin sayısını gösterelim
                                            for col, count in missing_counts.items():
                                                fecol2.write(f"{col}: {count} missing values were removed.")

                                            # Güncellenmiş DataFrame'i gösterelim
                                            fecol2.dataframe(p2_df)



                                    ###########################################################
                                    #### MISSING VALUES - Handling - Removing & Imputations // Numeric - Impute with Mean() Value
                                    ###########################################################
                                    if missing_bi_selected_feature == "Numeric - Impute with Mean() Value":

                                        ###########
                                        q11, q12 = fecol1.columns(2)
                                        if numbutcat_process_stage == "Yes":
                                            # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                            q11.write(f"Numeric but categoric argument is : {num_but_cat_arg}")
                                        else:
                                            # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                            num_but_cat_arg = q11.number_input(
                                                f"Enter a custom value for Numeric but Categoric Argument {counter}:",
                                                min_value=1,  # Minimum değer
                                                value=10,  # Varsayılan başlangıç değeri
                                                step=1,
                                                key=f"num_but_cat_arg_{counter}"
                                                # Benzersiz key her defasında farklı olur
                                            )
                                            q11.write(f"Numeric but categoric argument is set to : {num_but_cat_arg}")
                                        ###
                                        if catbutcar_process_stage == "Yes":
                                            # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                            q12.write(f"Categoric but cardinal argument is : {cat_but_car_arg}")

                                        else:
                                            # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                            cat_but_car_arg = q12.number_input(
                                                f"Enter a custom value for Categoric but Cardinal Argument {counter}:",
                                                min_value=1,  # Minimum değer
                                                value=20,  # Varsayılan başlangıç değeri
                                                step=1,
                                                key=f"cat_but_car_arg_{counter}"
                                                # Benzersiz key her defasında farklı olur
                                            )
                                            q12.write(f"Categoric but Cardinal argument is set to : {cat_but_car_arg}")

                                        p2_cat_cols, p2_num_cols, p2_cat_but_car, p2_num_but_cat = grab_col_names(p2_df,
                                                                                                                  num_but_cat_arg,
                                                                                                                  cat_but_car_arg)
                                        ###########


                                        na_cols = [col for col in p2_df.columns if p2_df[col].isnull().sum() > 0]


                                        # Numerical Imputation Methods
                                        na_cols_num = [col for col in na_cols if col in p2_num_cols]
                                        na_cols_cat = [col for col in na_cols if col in p2_cat_cols + p2_cat_but_car]

                                        # Mean Imputation with unique key
                                        selected_mean = fecol1.multiselect(
                                            "Select variables to impute with mean:",
                                            [col for col in na_cols_num],
                                            key=f"selected_mean_multiselect_{counter}"
                                        )

                                        if selected_mean:
                                            # Initialize a dictionary to hold the count of filled missing values for each selected variable
                                            filled_counts = {}
                                            fill_values = {}

                                            # Perform mean imputation and store the results
                                            for col in selected_mean:
                                                before_fill = p2_df[col].isna().sum()
                                                mean_value = p2_df[col].mean()
                                                p2_df[col].fillna(mean_value, inplace=True)
                                                after_fill = p2_df[col].isna().sum()
                                                filled_counts[col] = before_fill - after_fill
                                                fill_values[col] = mean_value

                                            # Show the count of filled missing values and the fill values for each column
                                            for col in selected_mean:
                                                fecol2.write(
                                                    f"{col}: {filled_counts[col]} missing values were filled with the mean value of {fill_values[col]:.2f}.",
                                                    key=f"filled_count_{col}_{counter}"
                                                )

                                            # Display the updated DataFrame with a unique key
                                            fecol2.dataframe(p2_df, key=f"updated_df_{counter}")



                                    ###########################################################
                                    #### MISSING VALUES - Handling - Removing & Imputations // Numeric - Impute with Mode() Value
                                    ###########################################################
                                    if missing_bi_selected_feature == "Numeric - Impute with Mode() Value":

                                        ###########
                                        q11, q12 = fecol1.columns(2)
                                        if numbutcat_process_stage == "Yes":
                                            # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                            q11.write(f"Numeric but categoric argument is : {num_but_cat_arg}")
                                        else:
                                            # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                            num_but_cat_arg = q11.number_input(
                                                f"Enter a custom value for Numeric but Categoric Argument {counter}:",
                                                min_value=1,  # Minimum değer
                                                value=10,  # Varsayılan başlangıç değeri
                                                step=1,
                                                key=f"num_but_cat_arg_{counter}"
                                                # Benzersiz key her defasında farklı olur
                                            )
                                            q11.write(f"Numeric but categoric argument is set to : {num_but_cat_arg}")
                                        ###
                                        if catbutcar_process_stage == "Yes":
                                            # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                            q12.write(f"Categoric but cardinal argument is : {cat_but_car_arg}")

                                        else:
                                            # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                            cat_but_car_arg = q12.number_input(
                                                f"Enter a custom value for Categoric but Cardinal Argument {counter}:",
                                                min_value=1,  # Minimum değer
                                                value=20,  # Varsayılan başlangıç değeri
                                                step=1,
                                                key=f"cat_but_car_arg_{counter}"
                                                # Benzersiz key her defasında farklı olur
                                            )
                                            q12.write(f"Categoric but Cardinal argument is set to : {cat_but_car_arg}")

                                        p2_cat_cols, p2_num_cols, p2_cat_but_car, p2_num_but_cat = grab_col_names(p2_df,
                                                                                                                  num_but_cat_arg,
                                                                                                                  cat_but_car_arg)
                                        ###########

                                        na_cols = [col for col in p2_df.columns if p2_df[col].isnull().sum() > 0]


                                        # Numerical Imputation Methods
                                        na_cols_num = [col for col in na_cols if col in p2_num_cols]
                                        na_cols_cat = [col for col in na_cols if col in p2_cat_cols + p2_cat_but_car]


                                        # Mod Imputation with unique key
                                        selected_mod = fecol1.multiselect(
                                            "Select variables to impute with mode():",
                                            [col for col in na_cols_num],
                                            key=f"selected_mod_multiselect_{counter}"
                                        )


                                        if selected_mod:
                                            # Initialize dictionaries to hold the count of filled missing values and the mode values for each selected variable
                                            filled_counts = {}
                                            mode_values = {}

                                            # Perform mode imputation and store the results
                                            for col in selected_mod:
                                                before_fill = p2_df[col].isna().sum()
                                                mode_value = p2_df[col].mode()[0]  # Find mode value for the column
                                                p2_df[col].fillna(mode_value,
                                                                  inplace=True)  # Fill missing values with mode
                                                after_fill = p2_df[col].isna().sum()
                                                filled_counts[col] = before_fill - after_fill
                                                mode_values[col] = mode_value

                                            # Show the count of filled missing values and the mode values for each column
                                            for col in selected_mod:
                                                fecol2.write(
                                                    f"{col}: {filled_counts[col]} missing values were filled with the mode value of {mode_values[col]}.",
                                                    key=f"filled_mode_{col}_{counter}"
                                                )

                                            # Display the updated DataFrame with a unique key
                                            fecol2.dataframe(p2_df, key=f"updated_df_mode_{counter}")







                                    ###########################################################
                                    #### MISSING VALUES - Handling - Removing & Imputations // Numeric - Impute with Median() Value
                                    ###########################################################
                                    if missing_bi_selected_feature == "Numeric - Impute with Median() Value":

                                        ###########
                                        q11, q12 = fecol1.columns(2)
                                        if numbutcat_process_stage == "Yes":
                                            # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                            q11.write(f"Numeric but categoric argument is : {num_but_cat_arg}")
                                        else:
                                            # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                            num_but_cat_arg = q11.number_input(
                                                f"Enter a custom value for Numeric but Categoric Argument {counter}:",
                                                min_value=1,  # Minimum değer
                                                value=10,  # Varsayılan başlangıç değeri
                                                step=1,
                                                key=f"num_but_cat_arg_{counter}"
                                                # Benzersiz key her defasında farklı olur
                                            )
                                            q11.write(f"Numeric but categoric argument is set to : {num_but_cat_arg}")
                                        ###
                                        if catbutcar_process_stage == "Yes":
                                            # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                            q12.write(f"Categoric but cardinal argument is : {cat_but_car_arg}")

                                        else:
                                            # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                            cat_but_car_arg = q12.number_input(
                                                f"Enter a custom value for Categoric but Cardinal Argument {counter}:",
                                                min_value=1,  # Minimum değer
                                                value=20,  # Varsayılan başlangıç değeri
                                                step=1,
                                                key=f"cat_but_car_arg_{counter}"
                                                # Benzersiz key her defasında farklı olur
                                            )
                                            q12.write(f"Categoric but Cardinal argument is set to : {cat_but_car_arg}")

                                        p2_cat_cols, p2_num_cols, p2_cat_but_car, p2_num_but_cat = grab_col_names(p2_df,
                                                                                                                  num_but_cat_arg,
                                                                                                                  cat_but_car_arg)
                                        ###########

                                        na_cols = [col for col in p2_df.columns if p2_df[col].isnull().sum() > 0]


                                        # Numerical Imputation Methods
                                        na_cols_num = [col for col in na_cols if col in p2_num_cols]
                                        na_cols_cat = [col for col in na_cols if col in p2_cat_cols + p2_cat_but_car]


                                        # Median Imputation with unique key
                                        selected_median = fecol1.multiselect(
                                            "Select variables to impute with median:",
                                            [col for col in na_cols_num],
                                            key=f"selected_median_multiselect_{counter}"
                                        )

                                        if selected_median:
                                            # Initialize dictionaries to hold the count of filled missing values and the median values for each selected variable
                                            filled_counts = {}
                                            median_values = {}

                                            # Perform median imputation and store the results
                                            for col in selected_median:
                                                before_fill = p2_df[col].isna().sum()
                                                median_value = p2_df[col].median()
                                                p2_df[col].fillna(median_value, inplace=True)
                                                after_fill = p2_df[col].isna().sum()
                                                filled_counts[col] = before_fill - after_fill
                                                median_values[col] = median_value

                                            # Show the count of filled missing values and the median values for each column
                                            for col in selected_median:
                                                fecol2.write(
                                                    f"{col}: {filled_counts[col]} missing values were filled with the median value of {median_values[col]}.",
                                                    key=f"filled_median_{col}_{counter}"
                                                )

                                            # Display the updated DataFrame with a unique key
                                            fecol2.dataframe(p2_df, key=f"updated_df_median_{counter}")


                                    ###########################################################
                                    #### MISSING VALUES - Handling - Removing & Imputations // Numeric - Impute with Custom Value
                                    ###########################################################
                                    if missing_bi_selected_feature == "Numeric - Impute with Custom Value":

                                        ###########
                                        q11, q12 = fecol1.columns(2)
                                        if numbutcat_process_stage == "Yes":
                                            # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                            q11.write(f"Numeric but categoric argument is : {num_but_cat_arg}")
                                        else:
                                            # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                            num_but_cat_arg = q11.number_input(
                                                f"Enter a custom value for Numeric but Categoric Argument {counter}:",
                                                min_value=1,  # Minimum değer
                                                value=10,  # Varsayılan başlangıç değeri
                                                step=1,
                                                key=f"num_but_cat_arg_{counter}"
                                                # Benzersiz key her defasında farklı olur
                                            )
                                            q11.write(f"Numeric but categoric argument is set to : {num_but_cat_arg}")
                                        ###
                                        if catbutcar_process_stage == "Yes":
                                            # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                            q12.write(f"Categoric but cardinal argument is : {cat_but_car_arg}")

                                        else:
                                            # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                            cat_but_car_arg = q12.number_input(
                                                f"Enter a custom value for Categoric but Cardinal Argument {counter}:",
                                                min_value=1,  # Minimum değer
                                                value=20,  # Varsayılan başlangıç değeri
                                                step=1,
                                                key=f"cat_but_car_arg_{counter}"
                                                # Benzersiz key her defasında farklı olur
                                            )
                                            q12.write(f"Categoric but Cardinal argument is set to : {cat_but_car_arg}")

                                        p2_cat_cols, p2_num_cols, p2_cat_but_car, p2_num_but_cat = grab_col_names(p2_df,
                                                                                                                  num_but_cat_arg,
                                                                                                                  cat_but_car_arg)
                                        ###########

                                        na_cols = [col for col in p2_df.columns if p2_df[col].isnull().sum() > 0]


                                        # Numerical Imputation Methods
                                        na_cols_num = [col for col in na_cols if col in p2_num_cols]
                                        na_cols_cat = [col for col in na_cols if col in p2_cat_cols + p2_cat_but_car]


                                        # Custom Value Imputation with unique key
                                        selected_custom = fecol1.multiselect(
                                            "Select variables to impute with custom value:",
                                            [col for col in na_cols_num],
                                            key=f"selected_custom_multiselect_{counter}"
                                        )

                                        # Dictionary to hold custom values for each selected variable
                                        custom_values = {}
                                        for col in selected_custom:
                                            custom_values[col] = fecol1.number_input(f"Enter custom value for {col}:",
                                                                                     value=0,
                                                                                     key=f"custom_value_{col}_{counter}")

                                        if selected_custom:
                                            # Initialize dictionaries to hold the count of filled missing values for each selected variable
                                            filled_counts = {}

                                            # Perform custom value imputation and store the results
                                            for col in selected_custom:
                                                before_fill = p2_df[col].isna().sum()
                                                custom_value = custom_values[col]
                                                p2_df[col].fillna(custom_value, inplace=True)
                                                after_fill = p2_df[col].isna().sum()
                                                filled_counts[col] = before_fill - after_fill

                                            # Show the count of filled missing values and the custom values for each column
                                            for col in selected_custom:
                                                fecol1.write(
                                                    f"{col}: {filled_counts[col]} missing values were filled with the custom value of {custom_values[col]}.",
                                                    key=f"filled_custom_{col}_{counter}"
                                                )

                                            # Display the updated DataFrame with a unique key
                                            fecol1.dataframe(p2_df, key=f"updated_df_custom_{counter}")





                                    ###########################################################
                                    #### MISSING VALUES - Handling - Removing & Imputations // Numeric - Impute with Random Values Between 2 Bounds
                                    ###########################################################
                                    if missing_bi_selected_feature == "Numeric - Impute with Random Values Between 2 Bounds":

                                        ###########
                                        q11, q12 = fecol1.columns(2)
                                        if numbutcat_process_stage == "Yes":
                                            # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                            q11.write(f"Numeric but categoric argument is : {num_but_cat_arg}")
                                        else:
                                            # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                            num_but_cat_arg = q11.number_input(
                                                f"Enter a custom value for Numeric but Categoric Argument {counter}:",
                                                min_value=1,  # Minimum değer
                                                value=10,  # Varsayılan başlangıç değeri
                                                step=1,
                                                key=f"num_but_cat_arg_{counter}"
                                                # Benzersiz key her defasında farklı olur
                                            )
                                            q11.write(f"Numeric but categoric argument is set to : {num_but_cat_arg}")
                                        ###
                                        if catbutcar_process_stage == "Yes":
                                            # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                            q12.write(f"Categoric but cardinal argument is : {cat_but_car_arg}")

                                        else:
                                            # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                            cat_but_car_arg = q12.number_input(
                                                f"Enter a custom value for Categoric but Cardinal Argument {counter}:",
                                                min_value=1,  # Minimum değer
                                                value=20,  # Varsayılan başlangıç değeri
                                                step=1,
                                                key=f"cat_but_car_arg_{counter}"
                                                # Benzersiz key her defasında farklı olur
                                            )
                                            q12.write(f"Categoric but Cardinal argument is set to : {cat_but_car_arg}")

                                        p2_cat_cols, p2_num_cols, p2_cat_but_car, p2_num_but_cat = grab_col_names(p2_df,
                                                                                                                  num_but_cat_arg,
                                                                                                                  cat_but_car_arg)
                                        ###########

                                        na_cols = [col for col in p2_df.columns if p2_df[col].isnull().sum() > 0]


                                        # Numerical Imputation Methods
                                        na_cols_num = [col for col in na_cols if col in p2_num_cols]
                                        na_cols_cat = [col for col in na_cols if col in p2_cat_cols + p2_cat_but_car]


                                        # Random Imputation with unique key
                                        selected_random = fecol1.multiselect(
                                            "Select variables to impute with random values:",
                                            [col for col in na_cols_num],
                                            key=f"selected_random_multiselect_{counter}"
                                        )

                                        random_bounds = {}
                                        value_type = fecol1.radio(
                                            "Choose value type for random values between two bounds:",
                                            ("Integer", "Float"),
                                            key=f"value_type_radio_{counter}"
                                        )

                                        for col in selected_random:
                                            lower_bound = fecol1.number_input(
                                                f"Enter the lower bound for {col}:",
                                                value=0.0,
                                                format="%.2f",
                                                key=f"lower_bound_{col}_{counter}"
                                            )
                                            upper_bound = fecol1.number_input(
                                                f"Enter the upper bound for {col}:",
                                                value=10.0,
                                                format="%.2f",
                                                key=f"upper_bound_{col}_{counter}"
                                            )
                                            if upper_bound <= lower_bound:
                                                fecol1.error(
                                                    f"The upper bound must be greater than the lower bound for {col}.")
                                            else:
                                                random_bounds[col] = (lower_bound, upper_bound)

                                        if selected_random:
                                            def fillna_with_random(df, bounds, columns, value_type):
                                                for col in columns:
                                                    lower_bound, upper_bound = bounds[col]
                                                    num_missing = df[col].isnull().sum()
                                                    if value_type == "Integer":
                                                        random_values = np.random.randint(lower_bound, upper_bound + 1,
                                                                                          num_missing)
                                                    else:
                                                        random_values = np.random.uniform(lower_bound, upper_bound,
                                                                                          num_missing)
                                                    df[col].fillna(
                                                        pd.Series(random_values, index=df[df[col].isnull()].index),
                                                        inplace=True)
                                                    fecol1.write(f"{col} column: {num_missing} missing values filled.",
                                                                 key=f"filled_{col}_{counter}")

                                            # Ensure random_bounds has the necessary data before calling the function
                                            if random_bounds:
                                                fillna_with_random(p2_df, random_bounds, selected_random, value_type)
                                                fecol2.write(
                                                    f"Imputation with random {value_type.lower()} values between bounds was done for {len(selected_random)} variables.",
                                                    key=f"imputation_result_{counter}"
                                                )
                                                fecol2.dataframe(p2_df, key=f"updated_df_random_{counter}")




                                    ###########################################################
                                    #### MISSING VALUES - Handling - Removing & Imputations // Numeric - Impute with Interpolation
                                    ###########################################################
                                    if missing_bi_selected_feature == "Numeric - Impute with Interpolation":

                                        ###########
                                        q11, q12 = fecol1.columns(2)
                                        if numbutcat_process_stage == "Yes":
                                            # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                            q11.write(f"Numeric but categoric argument is : {num_but_cat_arg}")
                                        else:
                                            # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                            num_but_cat_arg = q11.number_input(
                                                f"Enter a custom value for Numeric but Categoric Argument {counter}:",
                                                min_value=1,  # Minimum değer
                                                value=10,  # Varsayılan başlangıç değeri
                                                step=1,
                                                key=f"num_but_cat_arg_{counter}"
                                                # Benzersiz key her defasında farklı olur
                                            )
                                            q11.write(f"Numeric but categoric argument is set to : {num_but_cat_arg}")
                                        ###
                                        if catbutcar_process_stage == "Yes":
                                            # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                            q12.write(f"Categoric but cardinal argument is : {cat_but_car_arg}")

                                        else:
                                            # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                            cat_but_car_arg = q12.number_input(
                                                f"Enter a custom value for Categoric but Cardinal Argument {counter}:",
                                                min_value=1,  # Minimum değer
                                                value=20,  # Varsayılan başlangıç değeri
                                                step=1,
                                                key=f"cat_but_car_arg_{counter}"
                                                # Benzersiz key her defasında farklı olur
                                            )
                                            q12.write(f"Categoric but Cardinal argument is set to : {cat_but_car_arg}")

                                        p2_cat_cols, p2_num_cols, p2_cat_but_car, p2_num_but_cat = grab_col_names(p2_df,
                                                                                                                  num_but_cat_arg,
                                                                                                                  cat_but_car_arg)
                                        ###########

                                        na_cols = [col for col in p2_df.columns if p2_df[col].isnull().sum() > 0]


                                        # Numerical Imputation Methods
                                        na_cols_num = [col for col in na_cols if col in p2_num_cols]
                                        na_cols_cat = [col for col in na_cols if col in p2_cat_cols + p2_cat_but_car]


                                        # Random Imputation with unique key
                                        selected_interpolation = fecol1.multiselect(
                                            "Select variables to impute with interpolation:",
                                            [col for col in na_cols_num],
                                            key=f"selected_interpolation_multiselect_{counter}"
                                        )


                                        # Interpolation check
                                        if selected_interpolation:

                                            if p2_df.index.isna().sum() > 0:
                                                fecol1.error(
                                                    "There are NaN values in the index. Please handle these NaN values in the date column before applying interpolation."
                                                )
                                            else:
                                                for col in selected_interpolation:
                                                    before_fill = p2_df[col].isna().sum()

                                                    # Check if index is of datetime type for time-based interpolation
                                                    if isinstance(p2_df.index, pd.DatetimeIndex):
                                                        # Use time-based interpolation if the index is datetime
                                                        p2_df[col] = p2_df[col].interpolate(method='time')
                                                        fecol1.write(f"Time-based interpolation applied to {col}.")
                                                    else:
                                                        # Use linear interpolation if the index is not datetime
                                                        p2_df[col] = p2_df[col].interpolate(method='linear')
                                                        fecol1.write(f"Linear interpolation applied to {col}.")

                                                    after_fill = p2_df[col].isna().sum()

                                                    fecol2.write(
                                                        f"{col}: {before_fill - after_fill} missing values filled using interpolation.")

                                                # Display the updated DataFrame
                                                fecol2.dataframe(p2_df, key=f"updated_df_interpolation_{counter}")

                                        else:
                                            fecol2.write("Please select variables for interpolation.")





                                    ###########################################################
                                    #### MISSING VALUES - Handling - Removing & Imputations // Numeric - Impute with group based mean()/median()/mode() Value
                                    ###########################################################
                                    if missing_bi_selected_feature == "Numeric - Impute with group based mean()/median()/mode() Value":

                                        ###########
                                        q11, q12 = fecol1.columns(2)
                                        if numbutcat_process_stage == "Yes":
                                            # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                            q11.write(f"Numeric but categoric argument is : {num_but_cat_arg}")
                                        else:
                                            # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                            num_but_cat_arg = q11.number_input(
                                                f"Enter a custom value for Numeric but Categoric Argument {counter}:",
                                                min_value=1,  # Minimum değer
                                                value=10,  # Varsayılan başlangıç değeri
                                                step=1,
                                                key=f"num_but_cat_arg_{counter}"
                                                # Benzersiz key her defasında farklı olur
                                            )
                                            q11.write(f"Numeric but categoric argument is set to : {num_but_cat_arg}")
                                        ###
                                        if catbutcar_process_stage == "Yes":
                                            # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                            q12.write(f"Categoric but cardinal argument is : {cat_but_car_arg}")

                                        else:
                                            # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                            cat_but_car_arg = q12.number_input(
                                                f"Enter a custom value for Categoric but Cardinal Argument {counter}:",
                                                min_value=1,  # Minimum değer
                                                value=20,  # Varsayılan başlangıç değeri
                                                step=1,
                                                key=f"cat_but_car_arg_{counter}"
                                                # Benzersiz key her defasında farklı olur
                                            )
                                            q12.write(f"Categoric but Cardinal argument is set to : {cat_but_car_arg}")

                                        p2_cat_cols, p2_num_cols, p2_cat_but_car, p2_num_but_cat = grab_col_names(p2_df,
                                                                                                                  num_but_cat_arg,
                                                                                                                  cat_but_car_arg)
                                        ###########

                                        na_cols = [col for col in p2_df.columns if p2_df[col].isnull().sum() > 0]


                                        # Numerical Imputation Methods
                                        na_cols_num = [col for col in na_cols if col in p2_num_cols]
                                        na_cols_cat = [col for col in na_cols if col in p2_cat_cols + p2_cat_but_car]


                                        # Group vars Imputation with unique key
                                        method = fecol1.radio(
                                            "Select the method for filling with group-based mean/median/mode:**",
                                            ("Mean", "Median", "Mode"),
                                            key=f"group_based_method_{counter}"
                                        )

                                        # Group vars Imputation with unique key
                                        selected_grouped_vars = fecol1.multiselect(
                                            "Select variables to impute with group based mean()/mod()/median():",
                                            [col for col in na_cols_num],
                                            key=f"selected_grouped_vars_multiselect_{counter}"
                                        )


                                        grouping_cat_vars = {}
                                        for idx, col in enumerate(selected_grouped_vars):
                                            grouping_cat_vars[col] = fecol1.selectbox(
                                                f"Select a categorical variable to group by for {col}:",
                                                p2_cat_cols,
                                                key=f"grouping_cat_vars_{col}_{counter}_{idx}"
                                            )

                                        if selected_grouped_vars:
                                            fillna_summary = []  # To store summary information for each variable

                                            for col in selected_grouped_vars:
                                                group_col = grouping_cat_vars[col]

                                                # Calculate values based on the selected method
                                                if method == "Mean":
                                                    fill_values = p2_df.groupby(group_col)[col].mean()
                                                elif method == "Median":
                                                    fill_values = p2_df.groupby(group_col)[col].median()
                                                elif method == "Mode":
                                                    fill_values = p2_df.groupby(group_col)[col].apply(
                                                        lambda x: x.mode()[0])

                                                # Count missing values before imputation
                                                missing_before = p2_df[col].isnull().sum()

                                                # Create a DataFrame to count fillna operations per category
                                                fill_counts = {category: 0 for category in fill_values.index}

                                                # Impute missing values based on group values and count operations
                                                def fill_with_group_values(row):
                                                    if pd.isna(row[col]):
                                                        fill_counts[row[group_col]] += 1
                                                        return fill_values[row[group_col]]
                                                    else:
                                                        return row[col]

                                                p2_df[col] = p2_df.apply(fill_with_group_values, axis=1)

                                                # Calculate number of missing values after imputation
                                                missing_after = p2_df[col].isnull().sum()

                                                # Total fillna operations performed
                                                total_fillna_operations = missing_before - missing_after

                                                # Append summary
                                                fillna_summary.append({
                                                    'Variable': col,
                                                    'Total Fillna Operations': total_fillna_operations,
                                                    'Details': fill_counts
                                                })

                                                # Display results
                                                fecol1.write(
                                                    f"**Group-by {method.lower()} values for {col} by {group_col}:**")
                                                for category in fill_values.index:
                                                    # Count the number of filled values for each category
                                                    num_filled_by_category = \
                                                        p2_df[(p2_df[group_col] == category) & (
                                                            p2_df[col].notna())].shape[0]
                                                    fecol1.write(
                                                        f"{group_col} = {category}: {method} = {fill_values[category]}, Number of fillna operations = {num_filled_by_category}")
                                                fecol2.write(f"**Total fillna operations:** {total_fillna_operations}")

                                                for category, count in fill_counts.items():
                                                    fecol2.write(
                                                        f"{group_col} = {category}: Number of fillna operations = {count}")

                                            # Convert summary to DataFrame and display
                                            fillna_summary_df = pd.DataFrame(fillna_summary)
                                            fecol2.write("**Fillna Summary:**")
                                            fecol2.dataframe(fillna_summary_df)
                                            fecol2.write("**Current DataFrame:**")
                                            fecol2.dataframe(p2_df)





                                    ###########################################################
                                    #### MISSING VALUES - Handling - Removing & Imputations // Categoric - Impute with Mod() Value
                                    ###########################################################
                                    if missing_bi_selected_feature == "Categoric - Impute with Mode() Value":

                                        ###########
                                        q11, q12 = fecol1.columns(2)
                                        if numbutcat_process_stage == "Yes":
                                            # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                            q11.write(f"Numeric but categoric argument is : {num_but_cat_arg}")
                                        else:
                                            # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                            num_but_cat_arg = q11.number_input(
                                                f"Enter a custom value for Numeric but Categoric Argument {counter}:",
                                                min_value=1,  # Minimum değer
                                                value=10,  # Varsayılan başlangıç değeri
                                                step=1,
                                                key=f"num_but_cat_arg_{counter}"
                                                # Benzersiz key her defasında farklı olur
                                            )
                                            q11.write(f"Numeric but categoric argument is set to : {num_but_cat_arg}")
                                        ###
                                        if catbutcar_process_stage == "Yes":
                                            # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                            q12.write(f"Categoric but cardinal argument is : {cat_but_car_arg}")

                                        else:
                                            # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                            cat_but_car_arg = q12.number_input(
                                                f"Enter a custom value for Categoric but Cardinal Argument {counter}:",
                                                min_value=1,  # Minimum değer
                                                value=20,  # Varsayılan başlangıç değeri
                                                step=1,
                                                key=f"cat_but_car_arg_{counter}"
                                                # Benzersiz key her defasında farklı olur
                                            )
                                            q12.write(f"Categoric but Cardinal argument is set to : {cat_but_car_arg}")

                                        p2_cat_cols, p2_num_cols, p2_cat_but_car, p2_num_but_cat = grab_col_names(p2_df,
                                                                                                                  num_but_cat_arg,
                                                                                                                  cat_but_car_arg)
                                        ###########

                                        na_cols = [col for col in p2_df.columns if p2_df[col].isnull().sum() > 0]


                                        # Numerical Imputation Methods
                                        na_cols_num = [col for col in na_cols if col in p2_num_cols]
                                        na_cols_cat = [col for col in na_cols if col in p2_cat_cols]


                                        # Categoric Mod Imputation with unique key
                                        selected_cat_mode = fecol1.multiselect(
                                            "Select variables to impute with mode() value:",
                                            [col for col in na_cols_cat],
                                            key=f"selected_cat_mode_multiselect_{counter}"
                                        )

                                        if selected_cat_mode:
                                            for col in selected_cat_mode:
                                                # Mode imputation
                                                mode_value = p2_df[col].mode()[0]
                                                num_missing = p2_df[col].isnull().sum()
                                                p2_df[col].fillna(mode_value, inplace=True)

                                                # Output details
                                                fecol1.write(
                                                    f"{col} column: {num_missing} missing values filled with mode value '{mode_value}'.")

                                            fecol2.write(
                                                f"Mode imputation was done for {len(selected_cat_mode)} variables.")
                                            fecol2.write("**Current DataFrame:**")
                                            fecol2.dataframe(p2_df)


                                    ###########################################################
                                    #### MISSING VALUES - Handling - Removing & Imputations // Categoric - Impute with Custom Value
                                    ###########################################################
                                    if missing_bi_selected_feature == "Categoric - Impute with Custom String Value":

                                        ###########
                                        q11, q12 = fecol1.columns(2)
                                        if numbutcat_process_stage == "Yes":
                                            # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                            q11.write(f"Numeric but categoric argument is : {num_but_cat_arg}")
                                        else:
                                            # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                            num_but_cat_arg = q11.number_input(
                                                f"Enter a custom value for Numeric but Categoric Argument {counter}:",
                                                min_value=1,  # Minimum değer
                                                value=10,  # Varsayılan başlangıç değeri
                                                step=1,
                                                key=f"num_but_cat_arg_{counter}"
                                                # Benzersiz key her defasında farklı olur
                                            )
                                            q11.write(f"Numeric but categoric argument is set to : {num_but_cat_arg}")
                                        ###
                                        if catbutcar_process_stage == "Yes":
                                            # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                            q12.write(f"Categoric but cardinal argument is : {cat_but_car_arg}")

                                        else:
                                            # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                            cat_but_car_arg = q12.number_input(
                                                f"Enter a custom value for Categoric but Cardinal Argument {counter}:",
                                                min_value=1,  # Minimum değer
                                                value=20,  # Varsayılan başlangıç değeri
                                                step=1,
                                                key=f"cat_but_car_arg_{counter}"
                                                # Benzersiz key her defasında farklı olur
                                            )
                                            q12.write(f"Categoric but Cardinal argument is set to : {cat_but_car_arg}")

                                        p2_cat_cols, p2_num_cols, p2_cat_but_car, p2_num_but_cat = grab_col_names(p2_df,
                                                                                                                  num_but_cat_arg,
                                                                                                                  cat_but_car_arg)
                                        ###########

                                        na_cols = [col for col in p2_df.columns if p2_df[col].isnull().sum() > 0]


                                        # Numerical Imputation Methods
                                        na_cols_num = [col for col in na_cols if col in p2_num_cols]
                                        na_cols_cat = [col for col in na_cols if col in p2_cat_cols + p2_cat_but_car]


                                        # Categoric Mod Imputation with unique key
                                        selected_cat_custom = fecol1.multiselect(
                                            "Select variables to impute with custom string value:",
                                            [col for col in na_cols_cat],
                                            key=f"selected_cat_custom_multiselect_{counter}"
                                        )

                                        custom_cat_values = {}
                                        for col in selected_cat_custom:
                                            custom_cat_values[col] = fecol1.text_input(
                                                f"Enter custom string value for {col}:", value="N/A",
                                                key=f"custom_value_input_{col}_{counter}")

                                        if selected_cat_custom:
                                            for col in selected_cat_custom:
                                                # Custom string imputation
                                                num_missing = p2_df[col].isnull().sum()
                                                p2_df[col].fillna(custom_cat_values[col], inplace=True)

                                                # Output details
                                                fecol1.write(
                                                    f"{col} column: {num_missing} missing values filled with custom string '{custom_cat_values[col]}'.")

                                            fecol2.write(
                                                f"Custom string imputation was done for {len(selected_cat_custom)} variables.")
                                            fecol2.write("**Current DataFrame:**")
                                            fecol2.dataframe(p2_df)




                                    ###########################################################
                                    #### MISSING VALUES - Handling - Removing & Imputations // Categoric - Impute with Random String Value
                                    ###########################################################
                                    if missing_bi_selected_feature == "Categoric - Impute with Random String Value":

                                        ###########
                                        q11, q12 = fecol1.columns(2)
                                        if numbutcat_process_stage == "Yes":
                                            # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                            q11.write(f"Numeric but categoric argument is : {num_but_cat_arg}")
                                        else:
                                            # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                            num_but_cat_arg = q11.number_input(
                                                f"Enter a custom value for Numeric but Categoric Argument {counter}:",
                                                min_value=1,  # Minimum değer
                                                value=10,  # Varsayılan başlangıç değeri
                                                step=1,
                                                key=f"num_but_cat_arg_{counter}"
                                                # Benzersiz key her defasında farklı olur
                                            )
                                            q11.write(f"Numeric but categoric argument is set to : {num_but_cat_arg}")
                                        ###
                                        if catbutcar_process_stage == "Yes":
                                            # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                            q12.write(f"Categoric but cardinal argument is : {cat_but_car_arg}")

                                        else:
                                            # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                            cat_but_car_arg = q12.number_input(
                                                f"Enter a custom value for Categoric but Cardinal Argument {counter}:",
                                                min_value=1,  # Minimum değer
                                                value=20,  # Varsayılan başlangıç değeri
                                                step=1,
                                                key=f"cat_but_car_arg_{counter}"
                                                # Benzersiz key her defasında farklı olur
                                            )
                                            q12.write(f"Categoric but Cardinal argument is set to : {cat_but_car_arg}")

                                        p2_cat_cols, p2_num_cols, p2_cat_but_car, p2_num_but_cat = grab_col_names(p2_df,
                                                                                                                  num_but_cat_arg,
                                                                                                                  cat_but_car_arg)
                                        ###########

                                        na_cols = [col for col in p2_df.columns if p2_df[col].isnull().sum() > 0]


                                        # Numerical Imputation Methods
                                        na_cols_num = [col for col in na_cols if col in p2_num_cols]
                                        na_cols_cat = [col for col in na_cols if col in p2_cat_cols + p2_cat_but_car]


                                        # Categoric Mod Imputation with unique key
                                        selected_cat_random = fecol1.multiselect(
                                            "Select variables to impute with random string value:",
                                            [col for col in na_cols_cat],
                                            key=f"selected_cat_random_multiselect_{counter}"
                                        )

                                        random_cat_strings = {}
                                        for col in selected_cat_random:
                                            random_cat_strings[col] = fecol1.text_input(
                                                f"Enter random strings for {col} (separate by commas):", value="A,B,C",
                                                key=f"random_cat_input_{col}_{counter}"
                                            )

                                        if selected_cat_random:

                                            def fillna_with_random_cat(df, strings_dict, columns):
                                                # İndekslerde tekrarlayan değerler olup olmadığını kontrol et
                                                if df.index.duplicated().any():
                                                    fecol1.error(
                                                        "İşlem yapılamadı. DataFrame indekslerinde tekrarlayan değerler var.")
                                                    # Tekrarlayan indeksleri ekranda göster
                                                    duplicated_indices = df.index[df.index.duplicated()].unique()
                                                    fecol1.write("**Tekrarlayan indeksler:**")
                                                    fecol1.write(duplicated_indices)

                                                    return  # İşlem yapılmadan çıkış yapılır
                                                else:
                                                    for col in columns:
                                                        # Rastgele stringler arasından seçim yap
                                                        strings = strings_dict[col].split(",")
                                                        num_missing = df[col].isnull().sum()

                                                        if num_missing > 0:
                                                            # Eksik değerlerin indeksini al
                                                            missing_indices = df[df[col].isnull()].index

                                                            # Her eksik değer için rastgele string seç
                                                            random_choices = pd.Series(
                                                                np.random.choice(strings, num_missing),
                                                                index=missing_indices)

                                                            # fillna ile eksik değerleri doldur
                                                            df[col].fillna(random_choices, inplace=True)

                                                            # Doldurulan eksik değerlerin sayısı ve string listesi hakkında bilgi ver
                                                            fecol1.write(
                                                                f"{col}: {num_missing} missing values filled with random strings from {strings}."
                                                            )

                                            # Eksik değerleri doldur ve bu işlemi df'de kalıcı yap
                                            fillna_with_random_cat(p2_df, random_cat_strings, selected_cat_random)

                                            if not p2_df.index.duplicated().any():  # İşlem sadece indekslerde tekrar yoksa çalışır
                                                fecol1.write(
                                                    f"Random string imputation was done for {len(selected_cat_random)} variables.")
                                                # İşlem sonrası güncellenmiş DataFrame'i göster
                                                fecol1.dataframe(p2_df)



                                    ##############################
                                    #### MISSING VALUES - Handling - Handling - Removing & Imputations // kNN Imputation
                                    ##############################
                                    if missing_bi_selected_feature == "kNN Based Imputations":

                                        ###########
                                        q11, q12 = fecol1.columns(2)
                                        if numbutcat_process_stage == "Yes":
                                            # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                            q11.write(f"Numeric but categoric argument is : {num_but_cat_arg}")
                                        else:
                                            # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                            num_but_cat_arg = q11.number_input(
                                                f"Enter a custom value for Numeric but Categoric Argument {counter}:",
                                                min_value=1,  # Minimum değer
                                                value=10,  # Varsayılan başlangıç değeri
                                                step=1,
                                                key=f"num_but_cat_arg_{counter}"
                                                # Benzersiz key her defasında farklı olur
                                            )
                                            q11.write(f"Numeric but categoric argument is set to : {num_but_cat_arg}")
                                        ###
                                        if catbutcar_process_stage == "Yes":
                                            # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                            q12.write(f"Categoric but cardinal argument is : {cat_but_car_arg}")

                                        else:
                                            # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                            cat_but_car_arg = q12.number_input(
                                                f"Enter a custom value for Categoric but Cardinal Argument {counter}:",
                                                min_value=1,  # Minimum değer
                                                value=20,  # Varsayılan başlangıç değeri
                                                step=1,
                                                key=f"cat_but_car_arg_{counter}"
                                                # Benzersiz key her defasında farklı olur
                                            )
                                            q12.write(f"Categoric but Cardinal argument is set to : {cat_but_car_arg}")

                                        p2_cat_cols, p2_num_cols, p2_cat_but_car, p2_num_but_cat = grab_col_names(p2_df,
                                                                                                                  num_but_cat_arg,
                                                                                                                  cat_but_car_arg)
                                        ###########

                                        na_cols = [col for col in p2_df.columns if p2_df[col].isnull().sum() > 0]
                                        # Sayısal olmayan sütunları kontrol et
                                        non_numeric_cols_in_p2_num_cols = [col for col in p2_num_cols if
                                                                           not pd.api.types.is_numeric_dtype(
                                                                               p2_df[col])]
                                        non_numeric_cols_in_p2_cat_cols = [col for col in p2_cat_cols if
                                                                           not pd.api.types.is_numeric_dtype(
                                                                               p2_df[col])]

                                        # Eğer sayısal olmayan sütunlar varsa kullanıcıya uyarı ver ve işlemi durdur
                                        if non_numeric_cols_in_p2_num_cols or non_numeric_cols_in_p2_cat_cols:
                                            fecol1.warning(
                                                "Sayısal olmayan sütunlar bulundu! kNN imputation yalnızca sayısal değişkenler ile yapılabilir. "
                                                "Lütfen sayısal olmayan sütunları kontrol edin:"
                                            )
                                            if non_numeric_cols_in_p2_num_cols:
                                                fecol1.write(
                                                    f"Sayısal olmayan sütunlar (p2_num_cols): {non_numeric_cols_in_p2_num_cols}")
                                            if non_numeric_cols_in_p2_cat_cols:
                                                fecol1.write(
                                                    f"Sayısal olmayan sütunlar (p2_cat_cols): {non_numeric_cols_in_p2_cat_cols}")
                                        else:
                                            # Kullanıcıdan n_neighbors değerini alma
                                            n_neighbors = fecol1.number_input(
                                                "kNN için n_neighbors değerini girin:", min_value=1, value=5,
                                                key=f"n_neighbors_knn_{counter}"
                                            )

                                            # Tüm numeric sütunları birleştir
                                            all_numeric_cols = p2_num_cols + p2_cat_cols  # Zaten numeric olduğu kontrol edildiği için bu listeyi kullanıyoruz

                                            # Eksik değerlerin hangi satırlarda olduğunu belirlemek için index numaralarını saklama
                                            imputed_indices = {}
                                            for col in all_numeric_cols:
                                                if p2_df[col].isnull().sum() > 0:
                                                    imputed_indices[col] = p2_df[p2_df[col].isnull()].index.tolist()
                                                    fecol1.write(
                                                        f"**{col} değişkeni için eksik değerlerin doldurulacağı indeksler:**")
                                                    fecol1.write(imputed_indices[col])

                                            # KNN Imputer işlemini yalnızca sayısal değişkenler üzerinde yapıyoruz
                                            imputer = KNNImputer(n_neighbors=n_neighbors)
                                            knn_imputed_df = pd.DataFrame(
                                                imputer.fit_transform(p2_df[all_numeric_cols]),
                                                columns=all_numeric_cols)
                                            fecol1.write(
                                                f"KNN Imputer ({n_neighbors} komşu ile) kullanarak eksik değerler dolduruldu.")
                                            fecol1.write(knn_imputed_df)

                                            # Kullanıcıya hangi değişken için atama değerlerini kullanmak istediğini sor
                                            selected_cols_to_update = fecol1.multiselect(
                                                "Hangi değişken(ler) için atama değerlerini kullanmak istiyorsunuz?",
                                                all_numeric_cols,
                                                key=f"selected_cols_to_update_{counter}"
                                            )

                                            if selected_cols_to_update:
                                                for col in selected_cols_to_update:
                                                    p2_df[col].update(knn_imputed_df[col])
                                                    fecol1.write(
                                                        f"**{col} değişkenindeki eksik değerler güncellendi.**")

                                                fecol1.write("Güncellenmiş DataFrame:")
                                                fecol1.write(p2_df)
                                            else:
                                                fecol1.warning(
                                                    "Herhangi bir değişken seçilmedi, DataFrame üzerinde kalıcı bir değişiklik yapılmadı.")





                        ################################################################################################
                        ##################   STRUCTURE ÜST BAŞLIĞI
                        ################################################################################################

                        # Eğer Encoding seçilirse encoding seçenekleri sunuluyor
                        if category_choice == "Structure":
                            structure_selected_feature = fecol1.selectbox(
                                "Structure işlemi seçin:",
                                options=structure_feature_options,
                                key=f"structure_selectbox_{counter}_feature"
                            )


                            ##############################
                            #### STRUCTURE - Quick Remove Variable(s)
                            ##############################
                            # Quick Remove Variable işlemi seçilirse
                            if structure_selected_feature == "Quick Remove Variable(s)":

                                # Kullanıcıdan kaldırmak istediği sütunları seçmesini isteyelim
                                columns_to_remove = fecol1.multiselect("Silmek istediğiniz sütunları seçin:",
                                                                       p2_df.columns)

                                # Eğer kullanıcı sütun seçerse, bu sütunları DataFrame'den çıkaralım
                                if columns_to_remove:
                                    p2_df = p2_df.drop(columns=columns_to_remove)

                                    # İşlem sonrası güncel DataFrame'i gösterelim
                                    fecol2.write("Güncel DataFrame:")
                                    fecol2.write(p2_df)
                                else:
                                    st.write("Silmek için sütun seçilmedi.")




                            ##############################
                            #### STRUCTURE - Time Variable Processing
                            ##############################
                            # Time Variable Processing işlemi seçilirse
                            if structure_selected_feature == "Time Variable Processing":

                                # Zaman değişkeni olup olmadığını soralım
                                time_var_present = fecol1.radio(
                                    "Is there a time variable in the DataFrame?",
                                    ("No", "Yes"),
                                    key=f"time_var_present_{counter}"
                                )

                                if time_var_present == "Yes":
                                    # Zaman değişkenini seçelim
                                    selected_time_var = fecol1.selectbox(
                                        "Select the time variable:",
                                        p2_df.columns,
                                        key=f"time_var_select_{counter}"
                                    )

                                    # Zaman değişkeninin veri tipini kontrol edelim ve ekrana yazdıralım
                                    time_var_type = p2_df[selected_time_var].dtype
                                    fecol1.write(
                                        f"The data type of the selected variable '{selected_time_var}' is: {time_var_type}")

                                    # Veri tipi datetime değilse ve kullanıcı isterse datetime'a çevirelim
                                    if not pd.api.types.is_datetime64_any_dtype(p2_df[selected_time_var]):
                                        convert_to_datetime = fecol1.radio(
                                            f"'{selected_time_var}' is not a datetime type. Do you want to convert it?",
                                            ("No", "Yes"),
                                            key=f"convert_to_datetime_{counter}"
                                        )

                                        if convert_to_datetime == "Yes":
                                            try:
                                                # Convert to datetime
                                                p2_df[selected_time_var] = pd.to_datetime(p2_df[selected_time_var],
                                                                                          utc=True)
                                                fecol1.write(
                                                    f"'{selected_time_var}' has been successfully converted to datetime.")

                                                # After conversion or if already in datetime format, check for frequency
                                                if isinstance(p2_df[selected_time_var], pd.Series):
                                                    fecol1.write(
                                                        f"'{selected_time_var}' is now of type: {p2_df[selected_time_var].dtype}")

                                                # Check if the frequency is defined
                                                if isinstance(p2_df.index, pd.DatetimeIndex) and p2_df.index.freq:
                                                    fecol2.write(
                                                        f"The frequency of the current index is: {p2_df.index.freqstr}")
                                                else:
                                                    fecol2.write("The frequency is not defined.")

                                                # Ask if the user wants to define a frequency
                                                define_freq = fecol2.radio(
                                                    "Do you want to define a frequency for the time variable?",
                                                    ("No", "Yes"),
                                                    key=f"define_freq_{counter}"
                                                )

                                                if define_freq == "Yes":
                                                    # Allow the user to select frequency
                                                    freq_options = {
                                                        'Hourly': 'H',
                                                        'Minutely': 'T',
                                                        'Secondly': 'S',
                                                        'Millisecondly': 'L',
                                                        # 'L' represents milliseconds in Pandas frequency strings
                                                        'Daily': 'D',
                                                        'Business Day': 'B',
                                                        'Weekly': 'W',
                                                        'Monthly': 'M',
                                                        'Quarterly': 'Q',
                                                        'Yearly': 'Y'
                                                    }

                                                    selected_freq = fecol2.selectbox(
                                                        "Select the frequency for the time variable:",
                                                        list(freq_options.keys()),
                                                        key=f"freq_select_{counter}")
                                                    freq_code = freq_options[selected_freq]

                                                    fecol2.write(
                                                        f"Selected frequency for the time variable '{selected_time_var}': {freq_code}")

                                                    # Apply the frequency using to_period() without setting the variable as index
                                                    p2_df[selected_time_var] = pd.to_datetime(
                                                        p2_df[selected_time_var])  # Ensure it's in datetime format
                                                    p2_df[selected_time_var] = p2_df[selected_time_var].dt.to_period(
                                                        freq_code)  # Convert to period with selected frequency
                                                    fecol2.write(f"Frequency applied: {freq_code}")
                                                    fecol2.write(p2_df)

                                            except Exception as e:
                                                fecol1.write(f"An error occurred during conversion: {e}")

                                        else:
                                            fecol1.write(f"No conversion was applied to '{selected_time_var}'.")

                                        # Ask if the user wants to set the time variable as index
                                        set_as_index = fecol2.radio(
                                            f"Do you want to set '{selected_time_var}' as the index?",
                                            ("No", "Yes"),
                                            key=f"set_as_index_{counter}"
                                        )

                                        if set_as_index == "Yes":
                                            # Sadece seçilen zaman değişkenini indeks olarak ayarla, frekans tanımlaması yapma
                                            p2_df.set_index(selected_time_var, inplace=True)
                                            fecol2.write(f"'{selected_time_var}' has been set as the index.")
                                        else:
                                            fecol2.write(f"'{selected_time_var}' was not set as the index.")


                                # Display the DataFrame being processed in ptab1
                                st.write("Current DataFrame:")
                                st.write(p2_df)






                        ################################################################################################
                        ##################   OUTLIERS ÜST BAŞLIĞI
                        ################################################################################################

                        # Eğer Outliers seçilirse encoding seçenekleri sunuluyor
                        if category_choice == "Outliers":
                            outliers_selected_feature = fecol1.selectbox(
                                "Outliers işlemi seçin:",
                                options=outliers_feature_options,
                                key=f"outliers_selectbox_{counter}_feature"
                            )

                            ##############################
                            #### OUTLIERS - Outliers Detections & Handling
                            ##############################
                            # Outliers Detections & Handling işlemi seçilirse
                            if outliers_selected_feature == "Outliers Detections & Handling":

                                ###########
                                q11, q12 = fecol1.columns(2)
                                if numbutcat_process_stage == "Yes":
                                    # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                    q11.write(f"Numeric but categoric argument is : {num_but_cat_arg}")
                                else:
                                    # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                    num_but_cat_arg = q11.number_input(
                                        f"Enter a custom value for Numeric but Categoric Argument {counter}:",
                                        min_value=1,  # Minimum değer
                                        value=10,  # Varsayılan başlangıç değeri
                                        step=1,
                                        key=f"num_but_cat_arg_{counter}"
                                        # Benzersiz key her defasında farklı olur
                                    )
                                    q11.write(f"Numeric but categoric argument is set to : {num_but_cat_arg}")
                                ###
                                if catbutcar_process_stage == "Yes":
                                    # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                    q12.write(f"Categoric but cardinal argument is : {cat_but_car_arg}")

                                else:
                                    # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                    cat_but_car_arg = q12.number_input(
                                        f"Enter a custom value for Categoric but Cardinal Argument {counter}:",
                                        min_value=1,  # Minimum değer
                                        value=20,  # Varsayılan başlangıç değeri
                                        step=1,
                                        key=f"cat_but_car_arg_{counter}"
                                        # Benzersiz key her defasında farklı olur
                                    )
                                    q12.write(f"Categoric but Cardinal argument is set to : {cat_but_car_arg}")

                                p2_cat_cols, p2_num_cols, p2_cat_but_car, p2_num_but_cat = grab_col_names(p2_df,
                                                                                                          num_but_cat_arg,
                                                                                                          cat_but_car_arg)
                                ###########

                                ##############
                                ## Outliers Detection column (fecol1)
                                ##############

                                # Initialize lists to keep track of columns with and without outliers
                                outlier_cols = []
                                no_outlier_cols = []

                                # Create two columns for outlier detection results
                                feocol1, feocol2 = fecol1.columns(2)

                                # Ask the user if they want to perform outlier detection
                                perform_outlier_detection = feocol1.radio("Do you want to perform outlier detection?",
                                                                         ("No", "Yes"),
                                                                         key=f"perform_outlier_detection_{counter}")

                                if perform_outlier_detection == "Yes":
                                    # Ask user if they want to use default or custom values for q1 and q3
                                    use_defaults = feocol2.radio(
                                        "Do you want to use default (q1=0.25, q3=0.75) values?", ("Yes", "No"),
                                        key=f"use_defaults_{counter}")

                                    if use_defaults == "No":
                                        q1 = feocol2.number_input("Enter q1 value:", min_value=0.0, max_value=1.0,
                                                                 value=0.25, step=0.01, key=f"q1_value_{counter}")
                                        q3 = feocol2.number_input("Enter q3 value (must be greater than q1):",
                                                                 min_value=q1 + 0.01, max_value=1.0, value=0.75,
                                                                 step=0.01, key=f"q3_value_{counter}")
                                    else:
                                        q1, q3 = 0.25, 0.75

                                    feocol1.write("---")
                                    feocol2.write("---")


                                    # Initialize dictionaries to store low_limit and up_limit values
                                    low_limit_dict = {}
                                    up_limit_dict = {}

                                    for col in p2_num_cols:
                                        try:
                                            # Calculate outlier thresholds
                                            low_limit, up_limit = outlier_thresholds(p2_df, col, q1, q3)

                                            # Store the thresholds in dictionaries with systematic keys
                                            low_limit_dict[f"{col}_low_limit"] = low_limit
                                            up_limit_dict[f"{col}_up_limit"] = up_limit

                                            # Determine outliers
                                            has_outliers = check_outlier(p2_df, col, q1, q3)

                                            if has_outliers:
                                                with feocol1:
                                                    st.markdown(f"**{col}**")
                                                    st.write(f"- Minimum Value: `{p2_df[col].min()}`")
                                                    st.write(f"- Lower Limit (low_limit): `{low_limit}`")
                                                    st.write(f"- Upper Limit (up_limit): `{up_limit}`")
                                                    st.write(f"- Maximum Value: `{p2_df[col].max()}`")
                                                    st.markdown(f"**:red_circle: Outliers detected**")
                                                outlier_cols.append(col)
                                            else:
                                                with feocol2:
                                                    st.markdown(f"**{col}**")
                                                    st.write(f"- Minimum Value: `{p2_df[col].min()}`")
                                                    st.write(f"- Maximum Value: `{p2_df[col].max()}`")
                                                    st.markdown(f"**:white_check_mark: No outliers detected**")
                                                no_outlier_cols.append(col)

                                        except KeyError:
                                            fecol1.error(f"Column '{col}' not found in the DataFrame.")

                                    # Display lists of columns with and without outliers
                                    fecol1.markdown("---")
                                    summary_col1, summary_col2 = fecol1.columns(2)

                                    with summary_col1:
                                        st.write("Columns with Outliers:")
                                        st.write(outlier_cols)

                                    with summary_col2:
                                        st.write("Columns without Outliers:")
                                        st.write(no_outlier_cols)

                                ##############
                                ## Outliers Handling column (fecol2)
                                ##############

                                if perform_outlier_detection == "Yes":
                                    fecol2.write("Handle Outliers")
                                    if not outlier_cols:
                                        fecol2.warning(
                                            "There are no outliers! Go to Outlier Detection Settings section to adjust q1 and q3 parameters.")
                                    else:
                                        outlier_handling = fecol2.radio("Do you want to perform outlier handling?",
                                                                        ("No", "Yes"),
                                                                        key=f"outlier_handling_{counter}")

                                        if outlier_handling == "Yes":
                                            outlier_handling_method = fecol2.radio(
                                                "Choose the outlier handling method:",
                                                ("Remove the outliers", "Re-assignment with thresholds"),
                                                key=f"outlier_handling_method_{counter}")

                                            if outlier_handling_method == "Remove the outliers":
                                                # Display a multiselect widget with the list of variables from outlier_indices_dict
                                                selected_variables_remove = fecol2.multiselect(
                                                    "Select variables to remove outliers from:", options=outlier_cols,
                                                    default=outlier_cols, key=f"selected_variables_remove_{counter}")
                                                # Apply the remove_outlier function to the selected variables
                                                if selected_variables_remove:
                                                    for var in selected_variables_remove:
                                                        # Retrieve the limits from the dictionaries
                                                        low_limit = low_limit_dict[f"{var}_low_limit"]
                                                        up_limit = up_limit_dict[f"{var}_up_limit"]
                                                        p2_df, rows_removed = remove_outlier(p2_df, var, low_limit, up_limit)
                                                        fecol2.write(
                                                            f"{rows_removed} outliers have been removed from `{var}`.")

                                                    fecol2.success(
                                                        "Outliers have been removed from the selected variables.")
                                                    # Display the updated DataFrame
                                                    fecol2.markdown("---")
                                                    fecol2.subheader("Current DataFrame")
                                                    fecol2.write(p2_df)

                                                else:
                                                    fecol2.warning("No variables selected.")

                                            if outlier_handling_method == "Re-assignment with thresholds":
                                                # Display a multiselect widget with the list of variables from outlier_indices_dict
                                                selected_variables = fecol2.multiselect(
                                                    "Select variables to re-assign outliers:", options=outlier_cols,
                                                    default=outlier_cols, key=f"selected_variables_reassign_{counter}")
                                                if selected_variables:
                                                    reassign_count = {}  # Dictionary to keep track of re-assignments

                                                    for var in selected_variables:
                                                        # Retrieve the limits from the dictionaries
                                                        low_limit = low_limit_dict[f"{var}_low_limit"]
                                                        up_limit = up_limit_dict[f"{var}_up_limit"]

                                                        # Count the number of replacements before
                                                        before_replacement_count = \
                                                        p2_df[(p2_df[var] < low_limit) | (p2_df[var] > up_limit)].shape[
                                                            0]

                                                        # Call replace_with_thresholds with the appropriate limits
                                                        p2_df = replace_with_thresholds(p2_df, var, low_limit, up_limit)

                                                        # Count the number of replacements after
                                                        after_replacement_count = \
                                                        p2_df[(p2_df[var] < low_limit) | (p2_df[var] > up_limit)].shape[
                                                            0]

                                                        # Calculate the number of re-assignments
                                                        reassign_count[
                                                            var] = before_replacement_count - after_replacement_count

                                                    fecol2.success(
                                                        "Outliers have been re-assigned in the selected variables.")

                                                    # Display the re-assignment counts
                                                    fecol2.subheader("Re-assignment Counts")
                                                    for var, count in reassign_count.items():
                                                        fecol2.write(f"**{var}:** {count} re-assignments")

                                                    # Display the updated DataFrame
                                                    fecol2.markdown("---")
                                                    fecol2.subheader("Current DataFrame:")
                                                    fecol2.write(p2_df)

                                                else:
                                                    fecol2.warning("No variables selected.")

                                ##############
                                ## outlier detection for individual column (st)
                                ##############
                                st.subheader("Outlier içeren değişkenlerin diğer değişkenler ile birlikte incelenmesi:")
                                col_name = st.selectbox("Select the column to see outliers in the dataframe:", outlier_cols,
                                                        key=f"outlier_col_selectbox_{counter}")

                                if col_name:
                                    st.write(f"Detecting outliers in column: `{col_name}`")

                                    # Call the function
                                    outliers, outlier_index = grab_outliers(p2_df, col_name,
                                                                            low_limit_dict[f"{col_name}_low_limit"],
                                                                            up_limit_dict[f"{col_name}_up_limit"],
                                                                            index=True)

                                    # Count the number of outliers
                                    outlier_count = outliers.shape[0]

                                    # Display the count of outliers
                                    st.write(f"Number of outliers detected: `{outlier_count}`")

                                    # Add information about outliers
                                    st.markdown("The outliers with indices in the dataframe:")
                                    st.write(outliers)

                                else:
                                    st.write("Outlier detection has been skipped.")




                            ##############################
                            #### OUTLIERS - Local Outlier Factor Detections & Handling
                            ##############################
                            # Local Outlier Factor işlemi seçilirse
                            if outliers_selected_feature == "Local Outlier Factor & Handling":
                                fecol1.write("Local Outlier Factor (LOF):")

                                # Kullanıcıya LOF işlemi yapmak isteyip istemediğini sor (varsayılan olarak "Hayır")
                                perform_lof = fecol1.radio(
                                    "Local Outlier Factor (LOF) işlemini gerçekleştirmek istiyor musunuz?",
                                    ("Hayır", "Evet"), index=0, key=f"perform_lof_{counter}"
                                )

                                if perform_lof == "Evet":

                                    # Check for missing values
                                    na_cols = [col for col in p2_df.columns if p2_df[col].isnull().sum() > 0]

                                    if len(na_cols) > 0:
                                        # If missing values exist, notify the user and do not proceed with LOF
                                        fecol1.error(f"Missing values detected in the following columns: {na_cols}. "
                                                     "Please handle missing values in the 'Missing Values Handling' section before proceeding with LOF.")
                                    else:
                                        lof_df = p2_df.select_dtypes(include=[float, int])

                                        # Ask for the number of neighbors
                                        n_neighbors = fecol1.number_input(
                                            "Enter number of neighbors for LOF (default is 20):", min_value=1,
                                            value=20, step=1, key=f"n_neighbors_{counter}"
                                        )

                                        # Fit LOF model and get scores
                                        clf = LocalOutlierFactor(n_neighbors=n_neighbors)
                                        clf.fit_predict(lof_df)

                                        lof_df_scores = clf.negative_outlier_factor_

                                        # Show shapes
                                        fecol1.write(f"Shape of LOF scores: {lof_df_scores.shape[0]}")


                                        # Show the first five scores
                                        fecol1.write(f"First five LOF scores:")
                                        fecol1.write(pd.DataFrame(lof_df_scores[:5]).T)

                                        # Show the worst five scores
                                        worst_scores = np.sort(lof_df_scores)[:5]
                                        fecol1.write(
                                            "First five LOF scores after being sorted from largest to smallest scores (which means: worst five LOF scores):")
                                        fecol1.write(pd.DataFrame(worst_scores).T)

                                        # Ask for the number of worst scores to display
                                        elb = fecol1.number_input(
                                            "Enter the number of worst scores to display (default is 50):",
                                            min_value=1, value=50, step=1, key=f"elb_{counter}"
                                        )

                                        # Plot LOF graph
                                        # https://scikit-learn.org/dev/auto_examples/neighbors/plot_lof_outlier_detection.html
                                        ### eklenecek :)

                                        # Plot LOF scores
                                        scores = pd.DataFrame(np.sort(lof_df_scores))
                                        scores.plot(stacked=False, xlim=[0, elb], style='.-')
                                        plt.title('LOF Scores')
                                        plt.xlabel('Observations')
                                        plt.ylabel('LOF Score')
                                        fecol1.pyplot(plt)


                                        # Display the scores of the last `elb` observations
                                        fecol1.write(
                                            f"Scores of the worst last {elb} observations:")
                                        fecol1.write(pd.DataFrame(np.sort(lof_df_scores)[:elb]).T)  # DataFrame tablo olarak görüntülenir


                                        # Ask for threshold value
                                        fecol2.write("Determine which point on the graph you want to select:")
                                        th_elb = fecol2.number_input(
                                            "Enter threshold index value (default is 1):",
                                            min_value=0, value=1, step=1, key=f"th_elb_{counter}"
                                        )

                                        # Calculate threshold
                                        th_elb -= 1
                                        th = np.sort(lof_df_scores)[th_elb]

                                        # Display outliers below threshold
                                        outliers_df = lof_df[lof_df_scores < th]
                                        fecol2.write("Outliers based on threshold:")
                                        fecol2.dataframe(outliers_df)  # DataFrame'i transpoze ederek göster dedim hala yatay ! o nedenle aşağıdakini yazdım:

                                        # Show shape of the outliers DataFrame
                                        fecol2.write(f"Shape information of outliers: {outliers_df.shape[0]}")

                                        # Show descriptive statistics
                                        fecol2.write("Descriptive statistics of the DataFrame:")
                                        fecol2.write(lof_df.describe([0, 0.01, 0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99, 1]).T)

                                        # Ask user if they want to remove outliers
                                        remove_outliers = fecol2.radio(
                                            "Do you want to remove outliers based on LOF?", ("No", "Yes"),
                                            key=f"remove_outliers_{counter}"
                                        )

                                        if remove_outliers == "Yes":
                                            # Get indices of outliers
                                            outlier_indices_check = p2_df.index
                                            outlier_indices = outliers_df.index

                                            # Check for duplicate indices
                                            duplicate_indices = outlier_indices_check[outlier_indices_check.duplicated()]

                                            if len(duplicate_indices) > 0:
                                                # If there are duplicate indices, notify the user and ask for confirmation
                                                fecol2.warning(
                                                    "Dataframe Index'inde tekrar eden değerler var. Removing işleminde silme işlemi index numarasına göre yapıldığı için LOF analizi sonucu tespit edilen outlier'ların index'i ile aynı index'e sahip başka satırlar varsa onlar da silinecektir!"
                                                )

                                                # Ask the user if they want to proceed
                                                confirm_removal = fecol2.radio(
                                                    "Do you still want to proceed with removing these outliers?",
                                                    ("No", "Yes"), index=0, key=f"confirm_removal_{counter}"
                                                )

                                                if confirm_removal == "Yes":
                                                    # Remove outliers
                                                    p2_df = p2_df.drop(index=outlier_indices, axis=0)
                                                    fecol2.success(
                                                        "Outliers have been removed, including rows with duplicate indices.")
                                                    fecol2.write("Current Dataframe after removing outliers:")
                                                    fecol2.write(p2_df)
                                                else:
                                                    fecol2.warning(
                                                        "Outliers were not removed due to duplicate index issue.")
                                            else:
                                                # If there are no duplicate indices, remove outliers directly
                                                p2_df = p2_df.drop(index=outlier_indices, axis=0)
                                                fecol2.success("Outliers have been removed.")
                                                fecol2.write("Current Dataframe after removing outliers:")
                                                fecol2.write(p2_df)
                                        else:
                                            fecol2.warning("Outliers were not removed.")
                                            fecol2.write("The dataframe without handling:")
                                            fecol2.write(p2_df)









                        ################################################################################################
                        ##################   ENCODING ÜST BAŞLIĞI
                        ################################################################################################

                        # Eğer Encoding seçilirse encoding seçenekleri sunuluyor
                        if category_choice == "Encoding":
                            enc_selected_feature = fecol1.selectbox(
                                "Encoding işlemi seçin:",
                                options=enc_feature_options,
                                key=f"enc_selectbox_{counter}_feature"
                            )

                            ##############################
                            #### ENCODING - Label Encoding
                            ##############################
                            # Label Encoding işlemi seçilirse
                            if enc_selected_feature == "Label Encoding":

                                # Label Encoder fonksiyonu, NaN değerleri koruyacak şekilde güncellenmiş
                                def label_encoder_with_nan(dataframe, col, keep_nan=True):
                                    labelencoder = LabelEncoder()

                                    if keep_nan:
                                        # NaN değerlerin indekslerini sakla
                                        nan_mask = dataframe[col].isnull()

                                        # NaN olmayan değerlere label encoding uygulayalım
                                        encoded_values = labelencoder.fit_transform(dataframe.loc[~nan_mask, col])

                                        # Encode edilen değerleri orijinal dataframe'e yerleştirelim
                                        dataframe.loc[~nan_mask, col] = encoded_values

                                        # NaN değerleri koru
                                        dataframe.loc[nan_mask, col] = pd.NA
                                    else:
                                        # NaN değerler dahil, tüm değerlere label encoding uygulayalım
                                        encoded_values = labelencoder.fit_transform(dataframe[col])
                                        dataframe[col] = encoded_values

                                    return dataframe, labelencoder

                                # Kategorik sütunlar, numerik sütunlar ve kategorik ancak kardinal olanları alalım
                                ###########
                                q11, q12 = fecol1.columns(2)
                                if numbutcat_process_stage == "Yes":
                                    q11.write(f"Numeric but categoric argument is : {num_but_cat_arg}")
                                else:
                                    num_but_cat_arg = q11.number_input(
                                        f"Enter a custom value for Numeric but Categoric Argument {counter}:",
                                        min_value=1,  # Minimum değer
                                        value=10,  # Varsayılan başlangıç değeri
                                        step=1,
                                        key=f"num_but_cat_arg_{counter}"
                                    )
                                    q11.write(f"Numeric but categoric argument is set to : {num_but_cat_arg}")

                                if catbutcar_process_stage == "Yes":
                                    q12.write(f"Categoric but cardinal argument is : {cat_but_car_arg}")
                                else:
                                    cat_but_car_arg = q12.number_input(
                                        f"Enter a custom value for Categoric but Cardinal Argument {counter}:",
                                        min_value=1,  # Minimum değer
                                        value=20,  # Varsayılan başlangıç değeri
                                        step=1,
                                        key=f"cat_but_car_arg_{counter}"
                                    )
                                    q12.write(f"Categoric but Cardinal argument is set to : {cat_but_car_arg}")

                                p2_cat_cols, p2_num_cols, p2_cat_but_car, p2_num_but_cat = grab_col_names(p2_df,
                                                                                                          num_but_cat_arg,
                                                                                                          cat_but_car_arg)
                                ###########

                                # Kullanıcıya NaN değerler için bir seçenek sunalım
                                nan_handling_choice = fecol1.radio(
                                    "NaN değerleri nasıl ele almak istersiniz?",
                                    options=["NaN olarak kalsın", "Encode edilsin"],
                                    key=f"nan_handling_{counter}"
                                )

                                # Kullanıcıya p2_cat_cols listesinden birden fazla seçim yapma imkanı verelim
                                selected_cat_cols = fecol1.multiselect(
                                    "Label Encoding yapılacak kategorik değişkenleri seçin:",
                                    options=p2_cat_cols,
                                    key=f"cat_cols_multiselect_{counter}"
                                )

                                # Eğer kullanıcı seçim yapmışsa encoding işlemini uygulayalım
                                if selected_cat_cols:
                                    all_comparison_dfs = []  # Tüm unique değer karşılıklarını burada toplayacağız

                                    for col in selected_cat_cols:
                                        # NaN değerlerin korunması ya da encode edilmesi seçeneği
                                        keep_nan = True if nan_handling_choice == "NaN olarak kalsın" else False

                                        # Label Encoding uygulayalım ve LabelEncoder nesnesini alalım
                                        p2_df, le = label_encoder_with_nan(p2_df, col, keep_nan=keep_nan)

                                        # Unique değerleri ve bunların encode edilmiş hallerini alalım
                                        unique_values = p2_df[col].unique()
                                        non_nan_values = [val for val in unique_values if pd.notna(val)]
                                        original_values = le.inverse_transform([int(val) for val in non_nan_values])

                                        # Unique değerler ve encode edilmiş hallerini tablo şeklinde gösterelim
                                        if keep_nan:
                                            comparison_df = pd.DataFrame({
                                                'Column': col,
                                                'Original Values': list(original_values) + ['NaN'],
                                                'Encoded Values': list(non_nan_values) + [pd.NA]
                                            })
                                        else:
                                            comparison_df = pd.DataFrame({
                                                'Column': col,
                                                'Original Values': original_values,
                                                'Encoded Values': non_nan_values
                                            })

                                        all_comparison_dfs.append(comparison_df)  # Her bir tabloyu listeye ekleyelim

                                    # Tüm değişkenler için unique değer karşılıklarını tek bir tablo olarak gösterelim
                                    combined_comparison_df = pd.concat(all_comparison_dfs, ignore_index=True)

                                    fecol1.write(
                                        "Tüm değişkenler için Label Encoding sonrası unique değer karşılıkları:")
                                    fecol1.write(combined_comparison_df)

                                    # Son olarak güncel DataFrame'i yazdıralım
                                    st.write("Güncel DataFrame:")
                                    st.write(p2_df)

                                fecol1.write(f"Seçilen Encoding işlemi: {enc_selected_feature}")
                                selected_operations.append(enc_selected_feature)


                            ################################
                            #### ENCODING - One-Hot Encoding
                            ################################

                            # One-Hot Encoding işlemi seçilirse
                            elif enc_selected_feature == "One-Hot Encoding":

                                # One-Hot Encoder fonksiyonu
                                def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
                                    dataframe = pd.get_dummies(dataframe, columns=categorical_cols,
                                                               drop_first=drop_first)
                                    return dataframe

                                ## grab_col çağıralım:
                                ###########
                                q11, q12 = fecol1.columns(2)
                                if numbutcat_process_stage == "Yes":
                                    # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                    q11.write(f"Numeric but categoric argument is : {num_but_cat_arg}")
                                else:
                                    # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                    num_but_cat_arg = q11.number_input(
                                        f"Enter a custom value for Numeric but Categoric Argument {counter}:",
                                        min_value=1,  # Minimum değer
                                        value=10,  # Varsayılan başlangıç değeri
                                        step=1,
                                        key=f"num_but_cat_arg_{counter}"
                                        # Benzersiz key her defasında farklı olur
                                    )
                                    q11.write(f"Numeric but categoric argument is set to : {num_but_cat_arg}")
                                ###
                                if catbutcar_process_stage == "Yes":
                                    # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                    q12.write(f"Categoric but cardinal argument is : {cat_but_car_arg}")

                                else:
                                    # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                    cat_but_car_arg = q12.number_input(
                                        f"Enter a custom value for Categoric but Cardinal Argument {counter}:",
                                        min_value=1,  # Minimum değer
                                        value=20,  # Varsayılan başlangıç değeri
                                        step=1,
                                        key=f"cat_but_car_arg_{counter}"
                                        # Benzersiz key her defasında farklı olur
                                    )
                                    q12.write(f"Categoric but Cardinal argument is set to : {cat_but_car_arg}")

                                p2_cat_cols, p2_num_cols, p2_cat_but_car, p2_num_but_cat = grab_col_names(p2_df,
                                                                                                          num_but_cat_arg,
                                                                                                          cat_but_car_arg)
                                ###########

                                # Kullanıcıdan One-Hot Encoding yapılacak kategorik değişkenleri seçmesini isteyelim
                                selected_cat_cols = fecol1.multiselect(
                                    "One-Hot Encoding yapılacak kategorik değişkenleri seçin:",
                                    options=p2_cat_cols,
                                    key=f"one_hot_cat_cols_multiselect_{counter}"
                                )

                                if selected_cat_cols:

                                    # Seçilen değişkenlerde NaN değerlerin olup olmadığını kontrol edelim
                                    nan_cols = [col for col in selected_cat_cols if p2_df[col].isnull().sum() > 0]

                                    if len(nan_cols) > 0:
                                        # Eğer NaN değer varsa kullanıcıyı uyaralım ve işlemi durduralım
                                        fecol1.error(f"NaN values detected in the following columns: {nan_cols}. "
                                                     "Please handle missing values before proceeding with One-Hot Encoding to avoid issues with model accuracy.")
                                    else:
                                        # Kullanıcıya drop_first argümanını kullanmak isteyip istemediğini soralım
                                        drop_first_option = fecol1.checkbox("'drop_first=True' argümanını kullanmak istiyor musunuz?",
                                                                          key=f"drop_first_{counter}")

                                        # One-Hot Encoding işlemini uygulayalım
                                        p2_df = one_hot_encoder(p2_df, selected_cat_cols, drop_first=drop_first_option)

                                        # İşlemin tamamlandığına dair bilgi mesajı verelim
                                        fecol1.success("One-Hot Encoding işlemi başarıyla tamamlandı!")

                                        # Güncel DataFrame'i ekranda gösterelim
                                        fecol1.write("Güncel DataFrame:")
                                        fecol1.write(p2_df)



                            #############################
                            #### ENCODING - Rare Encoding
                            #############################

                            # Rare Encoding işlemi seçilirse
                            elif enc_selected_feature == "Rare Encoding":

                                # Kategorik sütunlar, numerik sütunlar ve kategorik ancak kardinal olanları alalım
                                ###########
                                q11, q12 = fecol1.columns(2)
                                if numbutcat_process_stage == "Yes":
                                    q11.write(f"Numeric but categoric argument is : {num_but_cat_arg}")
                                else:
                                    num_but_cat_arg = q11.number_input(
                                        f"Enter a custom value for Numeric but Categoric Argument {counter}:",
                                        min_value=1,  # Minimum değer
                                        value=10,  # Varsayılan başlangıç değeri
                                        step=1,
                                        key=f"num_but_cat_arg_{counter}"
                                    )
                                    q11.write(f"Numeric but categoric argument is set to : {num_but_cat_arg}")

                                if catbutcar_process_stage == "Yes":
                                    q12.write(f"Categoric but cardinal argument is : {cat_but_car_arg}")
                                else:
                                    cat_but_car_arg = q12.number_input(
                                        f"Enter a custom value for Categoric but Cardinal Argument {counter}:",
                                        min_value=1,
                                        value=20,
                                        step=1,
                                        key=f"cat_but_car_arg_{counter}"
                                    )
                                    q12.write(f"Categoric but Cardinal argument is set to : {cat_but_car_arg}")

                                p2_cat_cols, p2_num_cols, p2_cat_but_car, p2_num_but_cat = grab_col_names(p2_df,
                                                                                                          num_but_cat_arg,
                                                                                                          cat_but_car_arg)
                                ###########

                                # Kullanıcıya NaN değerlere encoding işlemi uygulanıp uygulanmayacağını soralım
                                apply_nan_encoding = fecol2.radio("NaN değerlere encoding işlemi uygulansın mı?",
                                                                  ("Evet", "Hayır"),
                                                                  key=f"apply_nan_encoding_{counter}")

                                # Rare Encoding analiz fonksiyonu
                                def rare_analyser(dataframe, cat_cols, target=None):
                                    results = {}
                                    for col in cat_cols:
                                        if target:
                                            analysis_df = pd.DataFrame({
                                                "COUNT": dataframe[col].value_counts(dropna=False),
                                                "RATIO": dataframe[col].value_counts(dropna=False) / len(dataframe),
                                                "TARGET_MEAN": dataframe.groupby(col)[target].mean()
                                            })
                                        else:
                                            analysis_df = pd.DataFrame({
                                                "COUNT": dataframe[col].value_counts(dropna=False),
                                                "RATIO": dataframe[col].value_counts(dropna=False) / len(dataframe)
                                            })
                                        results[col] = analysis_df
                                    return results


                                # Rare Encoder fonksiyonu - Orana göre
                                def rare_encoder_by_ratio(dataframe, categorical_cols, rare_perc, rare_label="Rare",
                                                          apply_nan=True):
                                    renc_df = dataframe.copy()
                                    for var in categorical_cols:
                                        # NaN değerlere encoding uygulanmayacaksa NaN'ları çıkarıyoruz
                                        if apply_nan == "Hayır":
                                            tmp = renc_df[var].dropna().value_counts(dropna=False) / len(
                                                renc_df.dropna(subset=[var]))
                                        else:
                                            tmp = renc_df[var].value_counts(dropna=False) / len(renc_df)
                                        rare_labels = tmp[tmp < rare_perc].index
                                        renc_df[var] = np.where(renc_df[var].isin(rare_labels), rare_label,
                                                                renc_df[var])
                                    return renc_df

                                # Rare Encoder fonksiyonu - Frekansa göre
                                def rare_encoder_by_freq(dataframe, categorical_cols, rare_freq, rare_label="Rare",
                                                         apply_nan=True):
                                    renc_df = dataframe.copy()
                                    for var in categorical_cols:
                                        # NaN değerlere encoding uygulanmayacaksa NaN'ları çıkarıyoruz
                                        if apply_nan == "Hayır":
                                            tmp = renc_df[var].dropna().value_counts(dropna=False)
                                        else:
                                            tmp = renc_df[var].value_counts(dropna=False)
                                        rare_labels = tmp[tmp < rare_freq].index
                                        renc_df[var] = np.where(renc_df[var].isin(rare_labels), rare_label,
                                                                renc_df[var])
                                    return renc_df

                                # Kullanıcıya hedef değişkeni seçip seçmek istemediğini soralım, default "Hayır" olarak ayarlandı
                                use_target_var = fecol1.radio(
                                    "Hedef değişkeni seçip target_mean görmek ister misiniz?",
                                    options=("Evet", "Hayır"),
                                    index=1,  # "Hayır" default olarak seçiliyor
                                    key=f"use_target_var_{counter}"
                                )

                                # Kullanıcıya hedef değişkeni seçtirelim (Eğer hedef değişkeni kullanmak istiyorsa)
                                if use_target_var == "Evet":
                                    target_var = fecol1.selectbox("Hedef değişkeni seçin:", options=p2_df.columns,
                                                                  key=f"rare_target_{counter}")
                                    rare_analysis_results = rare_analyser(p2_df, p2_cat_cols, target=target_var)
                                    for col, analysis_df in rare_analysis_results.items():
                                        fecol1.write(f"'{col}' değişkeni için rare encoding analizi (NaN dahil):")
                                        fecol1.write(analysis_df)
                                else:
                                    rare_analysis_results = rare_analyser(p2_df, p2_cat_cols)
                                    for col, analysis_df in rare_analysis_results.items():
                                        fecol1.write(f"'{col}' değişkeni için rare encoding analizi (NaN dahil):")
                                        fecol1.write(analysis_df)



                                # Kullanıcıdan rare encoding yapılacak değişkenleri seçmesini isteyelim
                                selected_cat_cols = fecol2.multiselect(
                                    "Rare Encoding yapılacak kategorik değişkenleri seçin:",
                                    options=p2_cat_cols,
                                    key=f"rare_cat_cols_multiselect_{counter}"
                                )

                                if selected_cat_cols:
                                    if len(selected_cat_cols) > 1:  # Birden fazla değişken seçildiyse ayrı ayar sorusu gösterilsin
                                        separate_settings = fecol2.checkbox(
                                            "Her bir değişken için ayrı encoding method, oran/frekans ve etiket tanımlamak istiyor musunuz?",
                                            key=f"separate_settings_{counter}"
                                        )
                                    else:
                                        separate_settings = False  # Sadece bir değişken seçildiyse otomatik olarak False olsun

                                    if separate_settings:
                                        # Her bir değişken için ayrı ayar
                                        encoding_method_dict = {}
                                        rare_perc_dict = {}
                                        rare_freq_dict = {}
                                        rare_label_dict = {}

                                        for col in selected_cat_cols:
                                            rarecol2_1, rarecol2_2, rarecol2_3 = fecol2.columns(3)

                                            encoding_method_dict[col] = rarecol2_1.radio(
                                                f"{col} için yöntem seçin:",
                                                options=["Orana göre", "Frekansa göre"],
                                                key=f"rare_encoding_method_{col}_{counter}"
                                            )

                                            if encoding_method_dict[col] == "Orana göre":
                                                rare_perc_dict[col] = rarecol2_2.number_input(
                                                    f"{col} için oran girin:",
                                                    min_value=0.0, max_value=1.0, step=0.01, value=0.01,
                                                    key=f"rare_perc_{col}_{counter}"
                                                )
                                                rare_label_dict[col] = rarecol2_3.text_input(
                                                    f"{col} için etiket girin:",
                                                    value="Rare", key=f"rare_label_{col}_{counter}"
                                                )
                                            elif encoding_method_dict[col] == "Frekansa göre":
                                                rare_freq_dict[col] = rarecol2_2.number_input(
                                                    f"{col} için frekans girin:",
                                                    min_value=1, step=1, value=10,
                                                    key=f"rare_freq_{col}_{counter}"
                                                )
                                                rare_label_dict[col] = rarecol2_3.text_input(
                                                    f"{col} için etiket girin:",
                                                    value="Rare", key=f"rare_label_{col}_{counter}"
                                                )

                                        # Seçilen encoding methoduna göre rare encoding işlemi yapalım
                                        for col in selected_cat_cols:
                                            if encoding_method_dict[col] == "Orana göre":
                                                p2_df = rare_encoder_by_ratio(p2_df, [col], rare_perc_dict[col],
                                                                              rare_label_dict[col], apply_nan_encoding)
                                            elif encoding_method_dict[col] == "Frekansa göre":
                                                p2_df = rare_encoder_by_freq(p2_df, [col], rare_freq_dict[col],
                                                                             rare_label_dict[col], apply_nan_encoding)
                                        fecol2.write(
                                            "Girilen oran/frekans değerinin altında kalan değişken sınıfları birleştirilmiştir.")

                                    else:
                                        # Tüm değişkenler için ortak ayar
                                        rarecol2_1, rarecol2_2, rarecol2_3 = fecol2.columns(3)

                                        encoding_method = rarecol2_1.radio(
                                            "Tüm değişkenler için yöntem seçin:",
                                            options=["Orana göre", "Frekansa göre"],
                                            key=f"rare_encoding_method_all_{counter}"
                                        )

                                        if encoding_method == "Orana göre":
                                            rare_perc = rarecol2_2.number_input(
                                                "Tüm değişkenler için oran girin:",
                                                min_value=0.0, max_value=1.0, step=0.01, value=0.01,
                                                key=f"rare_perc_all_{counter}"
                                            )
                                            rare_label = rarecol2_3.text_input(
                                                "Tüm değişkenler için etiket girin:",
                                                value="Rare", key=f"rare_label_all_{counter}"
                                            )
                                            p2_df = rare_encoder_by_ratio(p2_df, selected_cat_cols, rare_perc,
                                                                          rare_label, apply_nan_encoding)
                                        elif encoding_method == "Frekansa göre":
                                            rare_freq = rarecol2_2.number_input(
                                                "Tüm değişkenler için frekans girin:",
                                                min_value=1, step=1, value=10,
                                                key=f"rare_freq_all_{counter}"
                                            )
                                            rare_label = rarecol2_3.text_input(
                                                "Tüm değişkenler için etiket girin:",
                                                value="Rare", key=f"rare_label_all_{counter}"
                                            )
                                            p2_df = rare_encoder_by_freq(p2_df, selected_cat_cols, rare_freq,
                                                                         rare_label, apply_nan_encoding)

                                        fecol2.write(
                                            "Girilen oran/frekans değerinin altında kalan değişken sınıfları birleştirilmiştir.")

                                    # İşlemin tamamlandığına dair bilgi mesajı verelim
                                    fecol2.success("Rare Encoding işlemi başarıyla tamamlandı!")

                                    # Kullanıcıya Rare Analyser'ı tekrar çalıştırmak isteyip istemediğini soralım
                                    rerun_analyser = fecol2.radio(
                                        "Rare Analyser'ı tekrar çalıştırmak ister misiniz?",
                                        ("Hayır", "Evet"),
                                        key=f"rerun_analyser_{counter}"
                                    )

                                    if rerun_analyser == "Evet":
                                        # Rare Analyser'ı sadece rare encoding yapılmak üzere seçilen değişkenler için tekrar çalıştır
                                        fecol2.write("Seçilen değişkenler için Rare Encoding Analizi:")
                                        updated_analysis_results = rare_analyser(p2_df, selected_cat_cols,
                                                                                 target=target_var if use_target_var == "Evet" else None)

                                        for col, analysis_df in updated_analysis_results.items():
                                            fecol2.write(f"'{col}' değişkeni için rare encoding analizi (NaN dahil):")
                                            fecol2.write(analysis_df)

                                    # Güncel DataFrame'i ekranda gösterelim
                                    fecol2.write("Güncel DataFrame:")
                                    fecol2.write(p2_df)

                                else:
                                    fecol2.warning("Lütfen en az bir değişken seçin!")

                        ################################################################################################
                        ##################   SCALING ÜST BAŞLIĞI
                        ################################################################################################

                        # Eğer Scaling seçilirse encoding seçenekleri sunuluyor
                        if category_choice == "Scaling":
                            scaling_selected_feature = fecol1.selectbox(
                                "Scaling işlemi seçin:",
                                options=scaling_feature_options,
                                key=f"scaling_selectbox_{counter}_feature"
                            )


                            ##############################
                            #### Scaling - StandartScaler
                            ##############################
                            # Label Encoding işlemi seçilirse
                            if scaling_selected_feature == "StandardScaler":

                                # Kullanıcıdan Standard Scaling yapılacak numerik değişkenleri seçmesini isteyelim
                                selected_num_cols = fecol1.multiselect(
                                    "Standard Scaling yapılacak numerik değişkenleri seçin:",
                                    options=p2_df.select_dtypes(include=[float, int]).columns.tolist(),
                                    key=f"scaling_num_cols_multiselect_{counter}"
                                )

                                if selected_num_cols:
                                    for col in selected_num_cols:
                                        # Kullanıcıya yeni değişken türetmek isteyip istemediğini soralım
                                        new_column_option = fecol1.checkbox(
                                            f"{col} için yeni bir değişken türetmek ister misiniz?",
                                            key=f"new_col_{col}_{counter}")

                                        ss = StandardScaler()

                                        if new_column_option:
                                            # NEW_ önekini otomatik ekleyerek yeni değişken ismi alalım
                                            new_column_name_input = fecol1.text_input(
                                                f"Yeni {col} değişkeni için bir isim girin (otomatik olarak 'NEW_' ile başlayacak):",
                                                key=f"new_col_name_{col}_{counter}")
                                            new_column_name = f"NEW_{new_column_name_input}"

                                            # Yeni değişken türetme işlemi
                                            p2_df[new_column_name] = ss.fit_transform(p2_df[[col]])
                                            fecol1.success(
                                                f"Standard Scaling '{col}' için başarıyla '{new_column_name}' değişkeni oluşturuldu.")

                                            percentiles = fecol2.multiselect(
                                                "Özel yüzdelik dilimlerini seçin (varsayılan describe için boş bırakın):",
                                                options=[0, 0.01, 0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99, 1],
                                                key=f"percentiles_{col}_{counter}"
                                            )

                                            additional_compare_cols = fecol2.multiselect(
                                                "Başka değişkenlerle kıyaslamak ister misiniz? Seçin:",
                                                options=p2_df.columns.tolist(),
                                                key=f"compare_cols_{col}_{counter}"
                                            )

                                            if additional_compare_cols:
                                                comparison_columns = [col, new_column_name] + additional_compare_cols
                                                fecol2.write("Seçilen tüm değişkenler için özet istatistikler:")
                                                fecol2.write(p2_df[comparison_columns].describe(
                                                    percentiles if percentiles else None).T)
                                            else:
                                                fecol2.write(f"'{col}' ve '{new_column_name}' için özet istatistikler:")
                                                fecol2.write(p2_df[[col, new_column_name]].describe(
                                                    percentiles if percentiles else None).T)

                                        else:
                                            # Orijinal değişken üzerinde scaling işlemi
                                            p2_df[col] = ss.fit_transform(p2_df[[col]])
                                            fecol1.success(f"Standard Scaling '{col}' üzerinde başarıyla tamamlandı.")

                                    # Güncel DataFrame'i ekranda gösterelim
                                    fecol1.write("Güncel DataFrame:")
                                    fecol1.write(p2_df)

                            ##############################
                            #### Scaling - RobustScaler
                            ##############################
                            # Label Encoding işlemi seçilirse
                            if scaling_selected_feature == "RobustScaler":
                                # Kullanıcıdan Robust Scaling yapılacak numerik değişkenleri seçmesini isteyelim
                                selected_num_cols = fecol1.multiselect(
                                    "Robust Scaling yapılacak numerik değişkenleri seçin:",
                                    options=p2_df.select_dtypes(include=[float, int]).columns.tolist(),
                                    key=f"scaling_num_cols_multiselect_{counter}"
                                )

                                if selected_num_cols:
                                    for col in selected_num_cols:
                                        # Kullanıcıya yeni değişken türetmek isteyip istemediğini soralım
                                        new_column_option = fecol1.checkbox(
                                            f"{col} için yeni bir değişken türetmek ister misiniz?",
                                            key=f"new_col_{col}_{counter}")

                                        rs = RobustScaler()

                                        if new_column_option:
                                            # NEW_ önekini otomatik ekleyerek yeni değişken ismi alalım
                                            new_column_name_input = fecol1.text_input(
                                                f"Yeni {col} değişkeni için bir isim girin (otomatik olarak 'NEW_' ile başlayacak):",
                                                key=f"new_col_name_{col}_{counter}")
                                            new_column_name = f"NEW_{new_column_name_input}"

                                            # Yeni değişken türetme işlemi
                                            p2_df[new_column_name] = rs.fit_transform(p2_df[[col]])
                                            fecol1.success(
                                                f"Robust Scaling '{col}' için başarıyla '{new_column_name}' değişkeni oluşturuldu.")

                                            percentiles = fecol2.multiselect(
                                                "Özel yüzdelik dilimlerini seçin (varsayılan describe için boş bırakın):",
                                                options=[0, 0.01, 0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99, 1],
                                                key=f"percentiles_{col}_{counter}"
                                            )

                                            additional_compare_cols = fecol2.multiselect(
                                                "Başka değişkenlerle kıyaslamak ister misiniz? Seçin:",
                                                options=p2_df.columns.tolist(),
                                                key=f"compare_cols_{col}_{counter}"
                                            )

                                            if additional_compare_cols:
                                                comparison_columns = [col, new_column_name] + additional_compare_cols
                                                fecol2.write("Seçilen tüm değişkenler için özet istatistikler:")
                                                fecol2.write(p2_df[comparison_columns].describe(
                                                    percentiles if percentiles else None).T)
                                            else:
                                                fecol2.write(f"'{col}' ve '{new_column_name}' için özet istatistikler:")
                                                fecol2.write(p2_df[[col, new_column_name]].describe(
                                                    percentiles if percentiles else None).T)
                                        else:
                                            # Orijinal değişken üzerinde scaling işlemi
                                            p2_df[col] = rs.fit_transform(p2_df[[col]])
                                            fecol1.success(f"Robust Scaling '{col}' üzerinde başarıyla tamamlandı.")

                                    # Güncel DataFrame'i ekranda gösterelim
                                    fecol1.write("Güncel DataFrame:")
                                    fecol1.write(p2_df)


                            ##############################
                            #### Scaling - MinMAxScaler
                            ##############################
                            # Label Encoding işlemi seçilirse
                            if scaling_selected_feature == "MinMaxScaler":
                                # Kullanıcıdan MinMax Scaling yapılacak numerik değişkenleri seçmesini isteyelim
                                selected_num_cols = fecol1.multiselect(
                                    "MinMax Scaling yapılacak numerik değişkenleri seçin:",
                                    options=p2_df.select_dtypes(include=[float, int]).columns.tolist(),
                                    key=f"scaling_num_cols_multiselect_{counter}"
                                )

                                if selected_num_cols:
                                    for col in selected_num_cols:
                                        # Kullanıcıya yeni değişken türetmek isteyip istemediğini soralım
                                        new_column_option = fecol1.checkbox(
                                            f"{col} için yeni bir değişken türetmek ister misiniz?",
                                            key=f"new_col_{col}_{counter}")

                                        mms = MinMaxScaler()

                                        if new_column_option:
                                            # NEW_ önekini otomatik ekleyerek yeni değişken ismi alalım
                                            new_column_name_input = fecol1.text_input(
                                                f"Yeni {col} değişkeni için bir isim girin (otomatik olarak 'NEW_' ile başlayacak):",
                                                key=f"new_col_name_{col}_{counter}")
                                            new_column_name = f"NEW_{new_column_name_input}"

                                            # Yeni değişken türetme işlemi
                                            p2_df[new_column_name] = mms.fit_transform(p2_df[[col]])
                                            fecol1.success(
                                                f"MinMax Scaling '{col}' için başarıyla '{new_column_name}' değişkeni oluşturuldu.")


                                            # Yüzde dilimlerini seçme imkanı
                                            percentiles = fecol2.multiselect(
                                                "Özel yüzdelik dilimlerini seçin (varsayılan describe için boş bırakın):",
                                                options=[0, 0.01, 0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99, 1],
                                                key=f"percentiles_{col}_{counter}"
                                            )

                                            # Kullanıcının kıyaslamak istediği diğer değişkenleri seçme imkanı
                                            additional_compare_cols = fecol2.multiselect(
                                                "Başka değişkenlerle kıyaslamak ister misiniz? Seçin:",
                                                options=p2_df.columns.tolist(),
                                                key=f"compare_cols_{col}_{counter}"
                                            )

                                            # Ek değişken seçilmişse, hepsini tek tabloda göster
                                            if additional_compare_cols:
                                                comparison_columns = [col, new_column_name] + additional_compare_cols
                                                fecol2.write("Seçilen tüm değişkenler için özet istatistikler:")
                                                fecol2.write(p2_df[comparison_columns].describe(
                                                    percentiles if percentiles else None).T)
                                            else:
                                                # Ek değişken seçilmemişse sadece col ve new_column göster
                                                fecol2.write(f"'{col}' ve '{new_column_name}' için özet istatistikler:")
                                                fecol2.write(p2_df[[col, new_column_name]].describe(
                                                    percentiles if percentiles else None).T)


                                        else:
                                            # Orijinal değişken üzerinde scaling işlemi
                                            p2_df[col] = mms.fit_transform(p2_df[[col]])
                                            fecol1.success(f"MinMax Scaling '{col}' üzerinde başarıyla tamamlandı.")

                                    # Güncel DataFrame'i ekranda gösterelim
                                    fecol1.write("Güncel DataFrame:")
                                    fecol1.write(p2_df)



                        ################################################################################################
                        ##################   FEATURE EXTRACTION ÜST BAŞLIĞI
                        ################################################################################################

                        # Eğer Feature Extraction seçilirse feature extraction seçenekleri sunuluyor
                        elif category_choice == "Feature Extraction":
                            selected_feature = fecol1.selectbox(
                                "Feature extraction işlemi seçin:",
                                options=feature_options,
                                key=f"feature_selectbox_{counter}_feature"
                            )


                            #################################################################
                            ##################   FEATURE EXTRACTION - Time değişkeni türetmek
                            #################################################################

                            # Feature Extraction işlemleri (örnek olarak Time değişkeni türetmek ve diğerleri eklendi)
                            if selected_feature == "Time değişkeni türetmek":
                                # Zaman serisi içeren sütunlar ve indeks varsayımsız sunulacak
                                time_columns = ['index'] + list(p2_df.columns)

                                # Zaman serisi içeren sütunlardan veya indeksten birini seçelim
                                time_col = fecol1.selectbox("Zaman serisi içeren değişken veya indeks seçin:",
                                                          options=time_columns)

                                # Eğer zaman değişkeni index'te yer alıyorsa, index'i kullanarak işlem yapalım
                                if time_col == 'index':
                                    selected_data = p2_df.index
                                else:
                                    selected_data = p2_df[time_col]

                                # Eğer time_col bir PeriodIndex ise, to_timestamp() ile DatetimeIndex'e dönüştür
                                if isinstance(selected_data, pd.PeriodIndex):
                                    selected_data = selected_data.to_timestamp()

                                # Kullanıcıdan türetilecek yeni değişkenin label ismini alalım ve her seferinde counter'ı kullanarak dinamik hale getirelim
                                user_label = fecol1.text_input(f'Türetilecek yeni değişken için bir isim girin (NEW_):',
                                                             key=f"user_label_{counter}")
                                new_label = f"NEW_{user_label}"

                                # Kullanıcıya hangi özelliği türetmek istediğini dinamik olarak soralım
                                time_feature = fecol1.selectbox('Türetmek istediğiniz özelliği seçin:',
                                                              ['year', 'month', 'day', 'day_name', 'Hour', 'Minute',
                                                               'year_diff', 'month_diff'],
                                                              key=f'time_feature_{counter}')

                                # Seçilen özelliğe göre işlem yapalım
                                if time_feature == "year":
                                    p2_df[new_label] = selected_data.year
                                elif time_feature == "month":
                                    p2_df[new_label] = selected_data.month
                                elif time_feature == "day":
                                    p2_df[new_label] = selected_data.day
                                elif time_feature == "day_name":
                                    p2_df[new_label] = selected_data.day_name()
                                elif time_feature == "Hour":
                                    p2_df[new_label] = selected_data.hour + 1
                                elif time_feature == "Minute":
                                    p2_df[new_label] = selected_data.minute
                                elif time_feature == "year_diff":
                                    p2_df[new_label] = date.today().year - selected_data.year
                                elif time_feature == "month_diff":
                                    p2_df[new_label] = ((
                                                                    date.today().year - selected_data.year) * 12 + date.today().month - selected_data.month)

                                # Güncel DataFrame'i gösterelim
                                fecol1.write(f"Yeni {new_label} değişkeni türetildi. Güncel DataFrame:")
                                fecol1.write(p2_df)

                                # Yeni eklenen değişkenin unique değer sayısını kontrol edelim
                                unique_values = p2_df[new_label].unique()
                                num_unique_values = len(unique_values)

                                if num_unique_values == 1:
                                    # Tek unique değer varsa
                                    fecol2.write(f"Tek unique değer bulunmaktadır: {unique_values[0]}")

                                elif num_unique_values == 2:
                                    # İki unique değer varsa proportions_ztest testi yapılacak
                                    fecol2.write(f"İki unique değer bulunmaktadır. Değerler: {unique_values}")
                                    # Hedef değişken için kullanıcıdan seçim alalım
                                    # Hedef değişken için kullanıcıdan seçim alalım
                                    fe_target_var = fecol2.selectbox('Hedef değişkeni seçin:', options=p2_df.columns,
                                                                     key=f"fe_target_var_{counter}")

                                    # Yeni eklenen değişkenin seçilen hedef değişken üzerindeki sonuçlarına bakalım
                                    groupby_results = p2_df.groupby(new_label).agg({fe_target_var: "mean"})
                                    fecol2.write("Yeni değişkenin hedef değişken üzerindeki etkisi (Ortalama):")
                                    fecol2.write(groupby_results)

                                    # İstatistiki analiz: Oranlar testi (proportions_ztest) tüm unique değerler için yapılacak
                                    unique_values = p2_df[new_label].unique()

                                    for value in unique_values:
                                        count_1 = p2_df.loc[p2_df[new_label] == value, fe_target_var].sum()
                                        count_0 = p2_df.loc[p2_df[new_label] != value, fe_target_var].sum()
                                        nobs_1 = p2_df.loc[p2_df[new_label] == value, fe_target_var].shape[0]
                                        nobs_0 = p2_df.loc[p2_df[new_label] != value, fe_target_var].shape[0]

                                        if nobs_0 == 0 or nobs_1 == 0:  # Boş kümelerde test yapmamak için
                                            fecol2.write(f"Yeterli veri yok: {value} için test yapılmadı.")
                                            continue

                                        test_stat, pvalue = proportions_ztest(count=[count_1, count_0],
                                                                              nobs=[nobs_1, nobs_0])

                                        fecol2.write(
                                            f'{value} için Test Stat = {test_stat:.4f}, p-value = {pvalue:.4f}')
                                        if pvalue < 0.05:
                                            fecol2.write(
                                                f"H0 hipotezi reddedildi: {value} ile hedef değişken arasında fark vardır.")
                                        else:
                                            fecol2.write(
                                                f"H0 hipotezi reddedilemedi: {value} ile hedef değişken arasında fark yoktur.")

                                elif num_unique_values > 2:
                                    # İkiden fazla unique değer varsa ANOVA testi yapılacaktır yazısı gösterilecek
                                    fecol2.write(
                                        f"{num_unique_values} unique değer bulunmaktadır. ANOVA testi yapılacaktır.")




                            #####################################################################
                            ##################   FEATURE EXTRACTION - NaN bool değişkeni türetmek
                            #####################################################################

                            elif selected_feature == "NaN bool değişkeni türetmek":

                                # Kullanıcıdan bir değişken seçmesini isteyelim
                                selected_variable = fecol1.selectbox("NaN bool türetilecek değişkeni seçin:",
                                                                   options=p2_df.columns,
                                                                   key=f"nan_var_{counter}")

                                # Kullanıcıdan türetilecek yeni değişkenin label ismini alalım
                                user_label = fecol1.text_input(f'Türetilecek yeni değişken için bir isim girin (NEW_):',
                                                             key=f"nan_user_label_{counter}")

                                new_label = f"NEW_{user_label}"

                                # Yeni NaN bool değişkenini oluşturalım
                                p2_df[new_label] = p2_df[selected_variable].notnull().astype(int)

                                # Güncel DataFrame'i gösterelim
                                fecol1.write(f"NaN bool değişkeni türetildi. Güncel DataFrame:")
                                fecol1.write(p2_df)

                                # Yeni oluşturulan sütunun unique değer sayısını kontrol edelim
                                unique_values = p2_df[new_label].unique()
                                num_unique_values = len(unique_values)

                                if num_unique_values == 1:
                                    # Tek unique değer varsa işlem yapmayalım ve bilgi verelim
                                    fecol2.write(
                                        f"{new_label} değişkeninde tek unique değer bulunmaktadır, işlem yapılmadı.")

                                elif num_unique_values == 2:
                                    # İki unique değer varsa groupby ve proportions_ztest yapalım
                                    fecol2.write(
                                        f"{new_label} değişkeninde 2 unique değer bulunmaktadır. Groupby ve proportions test yapılacaktır.")

                                    fe_target_var = fecol2.selectbox('Hedef değişkeni seçin:', options=p2_df.columns,
                                                                     key=f"fe_target_var_{counter}")

                                    # Yeni eklenen değişkenin seçilen hedef değişken üzerindeki sonuçlarına bakalım
                                    groupby_results = p2_df.groupby(new_label).agg({fe_target_var: "mean"})
                                    fecol2.write("Yeni değişkenin hedef değişken üzerindeki etkisi (Ortalama):")
                                    fecol2.write(groupby_results)

                                    # İstatistiki analiz: Oranlar testi (proportions_ztest)
                                    count_1 = p2_df.loc[p2_df[new_label] == 1, fe_target_var].sum()
                                    count_0 = p2_df.loc[p2_df[new_label] == 0, fe_target_var].sum()
                                    nobs_1 = p2_df.loc[p2_df[new_label] == 1, fe_target_var].shape[0]
                                    nobs_0 = p2_df.loc[p2_df[new_label] == 0, fe_target_var].shape[0]

                                    test_stat, pvalue = proportions_ztest(count=[count_1, count_0],
                                                                          nobs=[nobs_1, nobs_0])

                                    fecol2.write(f'Test Stat = {test_stat:.4f}, p-value = {pvalue:.4f}')
                                    if pvalue < 0.05:
                                        fecol2.write(f"H0 hipotezi reddedildi: İki grup arasında fark vardır.")
                                    else:
                                        fecol2.write(f"H0 hipotezi reddedilemedi: İki grup arasında fark yoktur.")



                            ###########################################################################
                            ##################   FEATURE EXTRACTION - harf sayısı ile değişken türetmek
                            ###########################################################################

                            elif selected_feature == "Kategorik değişkenlerde harf sayısı ile değişken türetmek":

                                # Kategorik sütunlar, numerik sütunlar ve kategorik ancak kardinal olanları alalım
                                ###########
                                q11, q12 = fecol1.columns(2)
                                if numbutcat_process_stage == "Yes":
                                    # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                    q11.write(f"Numeric but categoric argument is : {num_but_cat_arg}")
                                else:
                                    # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                    num_but_cat_arg = q11.number_input(
                                        f"Enter a custom value for Numeric but Categoric Argument {counter}:",
                                        min_value=1,  # Minimum değer
                                        value=10,  # Varsayılan başlangıç değeri
                                        step=1,
                                        key=f"num_but_cat_arg_{counter}"
                                        # Benzersiz key her defasında farklı olur
                                    )
                                    q11.write(f"Numeric but categoric argument is set to : {num_but_cat_arg}")
                                ###
                                if catbutcar_process_stage == "Yes":
                                    # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                    q12.write(f"Categoric but cardinal argument is : {cat_but_car_arg}")

                                else:
                                    # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                    cat_but_car_arg = q12.number_input(
                                        f"Enter a custom value for Categoric but Cardinal Argument {counter}:",
                                        min_value=1,  # Minimum değer
                                        value=20,  # Varsayılan başlangıç değeri
                                        step=1,
                                        key=f"cat_but_car_arg_{counter}"
                                        # Benzersiz key her defasında farklı olur
                                    )
                                    q12.write(f"Categoric but Cardinal argument is set to : {cat_but_car_arg}")

                                p2_cat_cols, p2_num_cols, p2_cat_but_car, p2_num_but_cat = grab_col_names(p2_df,
                                                                                                          num_but_cat_arg,
                                                                                                          cat_but_car_arg)
                                ###########

                                # Kullanıcıdan bir değişken seçmesini isteyelim
                                selected_variable = fecol1.selectbox(
                                    "Harf sayısı (boşluklar harf olarak değerlendirilir) türetilecek kategorik değişkeni seçin:",
                                    options=p2_cat_cols + p2_cat_but_car, key=f"char_count_var_{counter}")

                                # Kullanıcıdan türetilecek yeni değişkenin label ismini alalım
                                user_label = fecol1.text_input(f'Türetilecek yeni değişken için bir isim girin (NEW_):',
                                                               key=f"char_count_user_label_{counter}")
                                new_label = f"NEW_{user_label}"

                                # Seçilen kategorik değişkenin harf sayısı ile yeni değişkeni oluşturalım
                                p2_df[new_label] = p2_df[selected_variable].str.len()

                                # Güncel DataFrame'i gösterelim
                                fecol1.write(f"{new_label} değişkeni türetildi. Güncel DataFrame:")
                                fecol1.write(p2_df)

                                # İşlem tamamlandı mesajını columns 2'ye yazdıralım
                                fecol2.write("İşlem tamamlandı.")



                            #############################################################################
                            ##################   FEATURE EXTRACTION - kelime sayısı ile değişken türetmek
                            #############################################################################

                            elif selected_feature == "Kategorik değişkenlerde kelime sayısı ile değişken türetmek":

                                # Kategorik sütunlar, numerik sütunlar ve kategorik ancak kardinal olanları alalım

                                ###########
                                q11, q12 = fecol1.columns(2)
                                if numbutcat_process_stage == "Yes":
                                    # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                    q11.write(f"Numeric but categoric argument is : {num_but_cat_arg}")
                                else:
                                    # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                    num_but_cat_arg = q11.number_input(
                                        f"Enter a custom value for Numeric but Categoric Argument {counter}:",
                                        min_value=1,  # Minimum değer
                                        value=10,  # Varsayılan başlangıç değeri
                                        step=1,
                                        key=f"num_but_cat_arg_{counter}"
                                        # Benzersiz key her defasında farklı olur
                                    )
                                    q11.write(f"Numeric but categoric argument is set to : {num_but_cat_arg}")
                                ###
                                if catbutcar_process_stage == "Yes":
                                    # Eğer kullanıcı 'Yes' seçerse, p2_df üzerinde grab_col_names fonksiyonu çağrılır
                                    q12.write(f"Categoric but cardinal argument is : {cat_but_car_arg}")

                                else:
                                    # Kullanıcıdan num_but_cat_arg değerini benzersiz key ile isteme
                                    cat_but_car_arg = q12.number_input(
                                        f"Enter a custom value for Categoric but Cardinal Argument {counter}:",
                                        min_value=1,  # Minimum değer
                                        value=20,  # Varsayılan başlangıç değeri
                                        step=1,
                                        key=f"cat_but_car_arg_{counter}"
                                        # Benzersiz key her defasında farklı olur
                                    )
                                    q12.write(f"Categoric but Cardinal argument is set to : {cat_but_car_arg}")

                                p2_cat_cols, p2_num_cols, p2_cat_but_car, p2_num_but_cat = grab_col_names(p2_df,
                                                                                                          num_but_cat_arg,
                                                                                                          cat_but_car_arg)
                                ###########

                                # Kullanıcıdan bir değişken seçmesini isteyelim
                                selected_variable = fecol1.selectbox("Kelime sayısı türetilecek kategorik değişkeni seçin:",
                                                                     options=p2_cat_cols + p2_cat_but_car, key=f"word_count_var_{counter}")

                                # Kullanıcıdan türetilecek yeni değişkenin label ismini alalım
                                user_label = fecol1.text_input(f'Türetilecek yeni değişken için bir isim girin (NEW_):',
                                                               key=f"word_count_user_label_{counter}")
                                new_label = f"NEW_{user_label}"

                                # Seçilen kategorik değişkenin kelime sayısı ile yeni değişkeni oluşturalım
                                p2_df[new_label] = p2_df[selected_variable].apply(lambda x: len(str(x).split(" ")))

                                # Güncel DataFrame'i gösterelim
                                fecol1.write(f"{new_label} değişkeni türetildi. Güncel DataFrame:")
                                fecol1.write(p2_df)

                                # İşlem tamamlandı mesajını columns 2'ye yazdıralım
                                fecol2.write("İşlem tamamlandı.")



                            ###########################################################################
                            ##################   FEATURE EXTRACTION - str içerenlerle değişken türetmek
                            ###########################################################################

                            elif selected_feature == "Belirli bir string ifade içerenlerle değişken türetmek":

                                # Kategorik sütunlar, numerik sütunlar ve kategorik ancak kardinal olanları alalım
                                p2_cat_cols, p2_num_cols, p2_cat_but_car, p2_num_but_cat = grab_col_names(p2_df)

                                # Kullanıcıdan bir değişken seçmesini isteyelim
                                selected_variable = fecol1.selectbox(
                                    "String ifade türetilecek kategorik değişkeni seçin:",
                                    options=p2_cat_cols + p2_cat_but_car, key=f"string_var_{counter}"
                                )

                                # Kullanıcıdan türetilecek yeni değişkenin label ismini alalım
                                user_label = fecol1.text_input(f'Türetilecek yeni değişken için bir isim girin (NEW_):',
                                                               key=f"string_user_label_{counter}")
                                new_label = f"NEW_{user_label}"

                                # Kullanıcıdan aranan string ifadeyi alalım
                                find_str = fecol1.text_input("Aradığınız string ifadesini girin:",
                                                             key=f"find_str_{counter}")

                                # Kullanıcıya üç seçenek sunalım
                                match_type = fecol1.selectbox(
                                    "Aradığınız string ifadenin nerede olmasını istiyorsunuz?",
                                    options=["... ile başlayan", "... ile biten", "herhangi bir yerinde ... içeren"],
                                    key=f"match_type_{counter}"
                                )

                                # Hedef değişkeni seçmesini isteyelim
                                target_var = fecol2.selectbox("Hedef değişkeni seçin:", options=p2_df.columns,
                                                              key=f"target_var_{counter}")

                                # Seçilen match_type'a göre uygun işlemi yapalım
                                if match_type == "... ile başlayan":
                                    p2_df[new_label] = p2_df[selected_variable].apply(
                                        lambda x: len([word for word in str(x).split() if word.startswith(find_str)])
                                    )
                                elif match_type == "... ile biten":
                                    p2_df[new_label] = p2_df[selected_variable].apply(
                                        lambda x: len([word for word in str(x).split() if word.endswith(find_str)])
                                    )
                                else:  # herhangi bir yerinde ... içeren
                                    p2_df[new_label] = p2_df[selected_variable].apply(
                                        lambda x: len([word for word in str(x).split() if find_str in word])
                                    )

                                # Güncel DataFrame'i gösterelim
                                fecol1.write(f"{new_label} değişkeni türetildi. Güncel DataFrame:")
                                fecol1.write(p2_df)

                                # Seçilen değişkene göre groupby işlemini yapalım
                                groupby_results = p2_df.groupby(new_label).agg({target_var: ['mean', 'count']})
                                fecol2.write(f"Yeni {new_label} değişkeninin grup ortalaması ve sayısı:")
                                fecol2.write(groupby_results)



                            #####################################################################
                            ##################   FEATURE EXTRACTION - Regex ile değişken türetmek
                            #####################################################################

                            elif selected_feature == "Regex ile metin analizi ile değişken türetmek":

                                # Kategorik sütunlar, numerik sütunlar ve kategorik ancak kardinal olanları alalım
                                p2_cat_cols, p2_num_cols, p2_cat_but_car, p2_num_but_cat = grab_col_names(p2_df)

                                # Kullanıcıdan bir değişken seçmesini isteyelim
                                selected_variable = fecol1.selectbox(
                                    "Regex ile analiz yapılacak kategorik değişkeni seçin:", options=p2_cat_cols + p2_cat_but_car,
                                    key=f"regex_var_{counter}")

                                # Kullanıcıdan türetilecek yeni değişkenin label ismini alalım
                                user_label = fecol1.text_input(f'Türetilecek yeni değişken için bir isim girin (NEW_):',
                                                               key=f"regex_user_label_{counter}")
                                new_label = f"NEW_{user_label}"

                                # Kullanıcıdan regex string ifadesi alalım, varsayılan olarak ([A-Za-z]+)\.
                                extract_str = fecol1.text_input("Aradığınız regex ifadesini girin:",
                                                                value=r"([A-Za-z]+)\.",
                                                                key=f"extract_str_{counter}")

                                # Hedef değişkeni seçmesini isteyelim
                                target_var = fecol2.selectbox("Hedef değişkeni seçin:", options=p2_df.columns,
                                                              key=f"target_var_{counter}")

                                # Seçilen değişkende regex ile eşleşen string'i çıkararak yeni değişkeni oluşturalım
                                p2_df[new_label] = p2_df[selected_variable].str.extract(f"{extract_str}", expand=False)

                                # Güncel DataFrame'i gösterelim
                                fecol1.write(f"{new_label} değişkeni türetildi. Güncel DataFrame:")
                                fecol1.write(p2_df)

                                # Fonksiyonun çalışma mantığını açıklayan bilgilendirme ve kod bloğu
                                explanation = """
                                Bu fonksiyon regex ifadesine göre seçilen bir metin değişkeninden belirli bir desen (pattern) yakalamak ve bu desenle yeni bir değişken türetmek için kullanılır.
                                Örneğin ([A-Za-z]+)\. regex ifadesi, harflerle başlayan ve nokta ile biten kelimeleri yakalar.
                                Aşağıdaki kod bloğu bu işlemi nasıl gerçekleştirdiğini göstermektedir:
                                """
                                fecol2.write(explanation)

                                # Kod bloğu
                                code_block = f"""
                                p2_df['{new_label}'] = p2_df['{selected_variable}'].str.extract(r"{extract_str}", expand=False)
                                """
                                fecol2.code(code_block, language='python')

                                # Seçilen değişkene göre groupby işlemini yapalım
                                groupby_results = p2_df.groupby(new_label).agg({target_var: ['count', 'mean']})
                                fecol2.write(f"Yeni {new_label} değişkeninin grup ortalaması ve sayısı:")
                                fecol2.write(groupby_results)


                            ####################################################################################
                            ##################   FEATURE EXTRACTION - Nümerik değişkeni bilinen sınırlara bölmek
                            ####################################################################################

                            elif selected_feature == "Nümerik değişkeni bilinen sınırlara bölerek değişken türetmek":
                                # Kullanıcıdan bir adet nümerik değişken seçmesini isteyelim
                                selected_numerical_var = fecol1.selectbox("Bir nümerik değişken seçin:",
                                                                          options=p2_df.select_dtypes(
                                                                              include=np.number).columns.tolist(),
                                                                          key=f"numerical_var_{counter}")

                                # Seçilen numerik değişkenin deskriptif tablosunu gösterelim
                                if selected_numerical_var:
                                    kkk = [0, 0.01, 0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99, 1]
                                    descriptive_stats = p2_df[selected_numerical_var].describe(percentiles=kkk)

                                    # Transpoze işleminden önce tabloyu DataFrame'e çevir ve satırları sütunlara dönüştür
                                    descriptive_stats_df = pd.DataFrame(descriptive_stats).transpose()

                                    fecol2.write(
                                        f"'{selected_numerical_var}' değişkeni için betimleyici istatistikler (yatay tablo):")

                                    # Tablonun genişliği ve hücre yüksekliklerini içeriğe göre otomatik ayarlamak için st.dataframe() kullan
                                    fecol2.dataframe(descriptive_stats_df)


                                # Kullanıcıdan türetilecek yeni değişkenin label ismini alalım
                                user_label = fecol1.text_input(f'Türetilecek yeni değişken için bir isim girin (NEW_):',
                                                               key=f"boundary_user_label_{counter}")
                                new_label = f"NEW_{user_label}"

                                # Kullanıcıya isteğe bağlı olarak başka bir değişken (nümerik veya kategorik) seçip bu değişkeni kullanarak filtreleme yapmak isteyip istemediğini soralım
                                add_filter = fecol1.checkbox("Başka bir değişkenle filtreleme yapmak ister misiniz?",
                                                             key=f"add_filter_{counter}")

                                if add_filter:
                                    # 4 sütunlu bir düzen oluşturalım: filtre, min, max, kategorik değer
                                    num_rows = fecol1.number_input("Kaç kategorik sınıfa bölmek istiyorsunuz?",
                                                                   min_value=1,
                                                                   step=1, key=f"num_rows_{counter}")
                                    filter_conditions = []

                                    for i in range(int(num_rows)):
                                        col1, col2, col3, col4, col5 = st.columns(
                                            5)  # 4 kolon oluştur: Filtre, Min, Max, Kategorik Değer

                                        # Filtreleme yapmak istediğiniz değişkeni ve koşulu girin
                                        with col1:
                                            selected_filter_var = col1.selectbox(f"Filtreleme için değişken {i + 1}:",
                                                                                 options=p2_df.columns.tolist(),
                                                                                 key=f"filter_var_{i}_{counter}")
                                            filter_value = col2.text_input(
                                                f"Filter {i + 1} ('male' or '> / < / == / >= / <= 50'):",
                                                key=f"filter_value_{i}_{counter}")

                                        # Min-Max değerleri girin
                                        with col3:
                                            min_value = col3.number_input(f"Min (excluded) (Sınıf {i + 1})",
                                                                          key=f"min_val_{i}_{counter}")
                                        with col4:
                                            max_value = col4.number_input(f"Max (included) (Sınıf {i + 1})",
                                                                          key=f"max_val_{i}_{counter}")

                                        # Kategorik değer girin
                                        with col5:
                                            class_label = col5.text_input(f"Kategorik Değer {i + 1}",
                                                                          key=f"class_val_{i}_{counter}")

                                        # Koşulları bir listeye ekleyelim
                                        if filter_value.startswith(
                                                (">", "<", "==", ">=", "<=")):  # Sayısal bir koşul ise
                                            filter_condition = f"(p2_df['{selected_filter_var}'] {filter_value})"
                                        else:  # Kategorik bir koşul (örn. 'male')
                                            filter_condition = f"(p2_df['{selected_filter_var}'] == '{filter_value}')"

                                        filter_conditions.append((filter_condition, min_value, max_value, class_label))

                                    # Sınıfları uygulayalım
                                    for condition, min_val, max_val, class_val in filter_conditions:
                                        # Eğer ek filtre varsa koşul o filtreyle birlikte uygulanacak
                                        p2_df.loc[(p2_df[selected_numerical_var] > min_val) & (
                                                p2_df[selected_numerical_var] <= max_val) & eval(
                                            condition), new_label] = class_val

                                else:
                                    # 3 sütunlu bir düzen: Min, Max ve Kategorik Değer
                                    num_classes = fecol1.number_input("Kaç kategorik sınıfa bölmek istiyorsunuz?",
                                                                      min_value=1, step=1, key=f"num_classes_{counter}")
                                    for i in range(int(num_classes)):
                                        col1, col2, col3 = st.columns(3)  # 3 kolon oluştur: Min, Max, Kategorik Değer

                                        # Min-Max ve Kategorik Değerleri girin
                                        with col1:
                                            min_value = col1.number_input(f"Min (excluded) (Sınıf {i + 1})",
                                                                          key=f"min_val_no_filter_{i}_{counter}")
                                        with col2:
                                            max_value = col2.number_input(f"Max (included) (Sınıf {i + 1})",
                                                                          key=f"max_val_no_filter_{i}_{counter}")
                                        with col3:
                                            class_label = col3.text_input(f"Kategorik Değer {i + 1}",
                                                                          key=f"class_val_no_filter_{i}_{counter}")

                                        # Koşulları uygulayalım
                                        p2_df.loc[(p2_df[selected_numerical_var] > min_value) & (
                                                p2_df[selected_numerical_var] <= max_value), new_label] = class_label

                                # Güncel DataFrame'i gösterelim
                                st.write(f"{new_label} değişkeni türetildi. Güncel DataFrame:")
                                st.write(p2_df)

                                # İşlem tamamlandı mesajını columns 2'ye yazdıralım
                                fecol2.write(
                                    "İşlem tamamlandı. Sınırlara göre kategorik değişken başarıyla oluşturuldu.")




                            ######################################################################################
                            ##################   FEATURE EXTRACTION - Nümerik değişkeni çeyreklik sınırlara bölmek
                            ######################################################################################

                            elif selected_feature == "Nümerik değişkeni çeyreklik sınırlara bölerek değişken türetmek":
                                # Kullanıcıdan bir adet nümerik değişken seçmesini isteyelim
                                selected_numerical_var = fecol1.selectbox("Bir nümerik değişken seçin:",
                                                                          options=p2_df.select_dtypes(
                                                                              include=np.number).columns.tolist(),
                                                                          key=f"numerical_var_{counter}")

                                # Seçilen numerik değişkenin deskriptif tablosunu gösterelim
                                if selected_numerical_var:
                                    kkk = [0, 0.01, 0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99, 1]
                                    descriptive_stats = p2_df[selected_numerical_var].describe(percentiles=kkk)

                                    # Transpoze edilmiş tabloyu gösterelim ve genişlik/yükseklik hücrelere göre ayarlansın
                                    descriptive_stats_df = pd.DataFrame(descriptive_stats).transpose()

                                    fecol2.write(
                                        f"'{selected_numerical_var}' değişkeni için betimleyici istatistikler (yatay tablo):")
                                    fecol2.dataframe(descriptive_stats_df)  # Dinamik boyutlu tablo
                                    

                                # Kullanıcıdan türetilecek yeni değişkenin label ismini alalım
                                user_label = fecol1.text_input(f'Türetilecek yeni değişken için bir isim girin (NEW_):',
                                                               key=f"boundary_user_label_{counter}")
                                new_label = f"NEW_{user_label}"

                                # Kullanıcıya birden fazla filtreleme yapma seçeneği sunalım
                                add_filter = fecol1.checkbox("Başka bir değişkenle filtreleme yapmak ister misiniz?",
                                                             key=f"add_filter_{counter}")

                                if add_filter:
                                    num_filters = fecol1.number_input("Kaç filtre eklemek istiyorsunuz?", min_value=1,
                                                                      step=1, key=f"num_filters_{counter}")
                                    filter_conditions = []

                                    # Filtre koşullarını toplayalım
                                    for i in range(int(num_filters)):
                                        filter_col1, filter_col2, filter_col3 = st.columns(
                                            3)  # Her filtre için üç sütun: değişken, koşul, önek
                                        with filter_col1:
                                            selected_filter_var = filter_col1.selectbox(
                                                f"Filtreleme yapmak istediğiniz değişken {i + 1}:",
                                                options=p2_df.columns.tolist(), key=f"filter_var_{i}_{counter}")
                                        with filter_col2:
                                            filter_value = filter_col2.text_input(
                                                f"{selected_filter_var} için filtre değeri girin (örn. 'male' veya '>= 50'):",
                                                key=f"filter_value_{i}_{counter}")
                                        with filter_col3:
                                            filter_prefix = filter_col3.text_input(
                                                f"{selected_filter_var} için önek girin (örn. 'female_'):",
                                                key=f"filter_prefix_{i}_{counter}")

                                        # Koşulun doğru oluşturulması
                                        if filter_value.startswith(
                                                (">", "<", "==", ">=", "<=")):  # Sayısal bir koşul ise
                                            filter_condition = f"(p2_df['{selected_filter_var}'] {filter_value})"
                                        else:  # Kategorik bir koşul (örn. 'male')
                                            filter_condition = f"(p2_df['{selected_filter_var}'] == '{filter_value}')"

                                        filter_conditions.append(
                                            (selected_filter_var, filter_value, filter_condition, filter_prefix))

                                    combined_filter_results = pd.Series([None] * len(p2_df), index=p2_df.index)

                                    # Her filtre için işlemleri birleştirerek tek bir sütuna aktaralım
                                    for i, (filter_var, filter_val, condition, prefix) in enumerate(filter_conditions):
                                        # Filtrelenmiş dataframe oluştur
                                        filtered_df = p2_df.loc[eval(condition)]

                                        # Filtrelenmiş verinin boş olup olmadığını kontrol edelim
                                        if filtered_df.empty:
                                            fecol2.write(
                                                f"Filtreleme sonrası '{filter_var}' ile '{filter_val}' koşulu sonucu boş bir veri seti oluştu.")
                                        else:
                                            # Çeyrek sınırlara göre qcut kullanarak kategorik değişken türetelim
                                            num_classes = 4  # Çeyreklik dilimleri 4'e bölelim
                                            quartile_labels = ['1. Çeyrek', '2. Çeyrek', '3. Çeyrek', '4. Çeyrek']

                                            # `qcut` işlemi ve sınırların yakalanması
                                            filtered_qcut, bin_edges = pd.qcut(filtered_df[selected_numerical_var],
                                                                               q=num_classes, labels=quartile_labels,
                                                                               retbins=True)

                                            # Sınırları ekrana yazdıralım
                                            fecol2.write(
                                                f"{filter_var} ile {filter_val} koşulu sonucu {new_label} için qcut sınırları: {bin_edges}")

                                            # Her çeyrek için gözlem sayısını yazdıralım
                                            for quartile in quartile_labels:
                                                count_in_quartile = (filtered_qcut == quartile).sum()
                                                fecol2.write(f"{quartile}: {count_in_quartile} gözlem")

                                            # Kullanıcının tanımladığı önek ile qcut sonuçlarını birleştirelim
                                            filtered_qcut_with_prefix = prefix + filtered_qcut.astype(str)

                                            # Filtre sonuçlarını birleştirip tek sütuna aktaralım
                                            combined_filter_results.loc[filtered_df.index] = filtered_qcut_with_prefix

                                    # Orijinal dataframe'e yeni sütunu ekleyelim
                                    p2_df[new_label] = combined_filter_results

                                else:
                                    # Çeyrek sınırlara göre qcut kullanarak kategorik değişken türetelim
                                    num_classes = 4  # Çeyreklik dilimleri 4'e bölelim
                                    quartile_labels = ['1. Çeyrek', '2. Çeyrek', '3. Çeyrek', '4. Çeyrek']

                                    # `qcut` işlemi ve sınırların yakalanması
                                    p2_df[new_label], bin_edges = pd.qcut(p2_df[selected_numerical_var], q=num_classes,
                                                                          labels=quartile_labels, retbins=True)

                                    # Sınırları ekrana yazdıralım
                                    fecol2.write(f"{new_label} için qcut sınırları: {bin_edges}")

                                    # Her çeyrek için gözlem sayısını yazdıralım
                                    for quartile in quartile_labels:
                                        count_in_quartile = (p2_df[new_label] == quartile).sum()
                                        fecol2.write(f"{quartile}: {count_in_quartile} gözlem")

                                # Güncel DataFrame'i gösterelim
                                st.write(f"{new_label} değişkeni türetildi. Güncel DataFrame:")
                                st.write(p2_df)

                                # İşlem tamamlandı mesajını columns 2'ye yazdıralım
                                fecol2.write(
                                    "İşlem tamamlandı. Çeyreklik sınırlara göre kategorik değişken başarıyla oluşturuldu.")



                            ###################################################################
                            ##################   FEATURE EXTRACTION - Matematiksel koşullar ile
                            ###################################################################

                            if selected_feature == "Koşullu matematiksel operasyonlar ile değişken türetmek":
                                # Kullanıcıdan kaç koşul gireceğini soralım
                                num_conditions = st.number_input("Kaç adet koşul girmek istiyorsunuz?", min_value=1,
                                                                 max_value=10, step=1, key="num_conditions")

                                # Kullanıcıdan yeni değişken ismi alalım (NEW_ otomatik olarak eklenecek)
                                user_label = fecol1.text_input(f'Türetilecek yeni değişken için bir isim girin (NEW_):',
                                                               key=f"user_label_{counter}")
                                new_label = f"NEW_{user_label}"

                                # Koşul, true ve false değerlerini alalım
                                conditions = []
                                for i in range(num_conditions):
                                    fecol1, fecol2, fecol3 = st.columns(3)

                                    with fecol1:
                                        condition = st.text_input(
                                            f"{i + 1}. Koşul girin (örn. '(p2_df[\"SibSp\"] + p2_df[\"Parch\"]) > 0')",
                                            key=f"condition_{counter}_{i}")

                                    if num_conditions == 1:
                                        # Eğer tek koşul varsa True ve False değerlerini ayrı ayrı alalım
                                        with fecol2:
                                            true_value = fecol2.text_input(
                                                f"{i + 1}. Koşul doğruysa (True) verilecek değer:",
                                                key=f"true_value_{counter}_{i}")
                                        with fecol3:
                                            false_value = fecol3.text_input(
                                                f"{i + 1}. Koşul yanlışsa (False) verilecek değer:",
                                                key=f"false_value_{counter}_{i}")
                                        conditions.append((condition, true_value, false_value))

                                    elif num_conditions > 1:
                                        # Eğer birden fazla koşul varsa her koşul için tek bir değer alalım
                                        with fecol2:
                                            result_value = fecol2.text_input(f"{i + 1}. Koşul için verilecek değer:",
                                                                             key=f"result_value_{counter}_{i}")
                                        # Her koşul için kullanıcıdan birer adet değer alıyoruz
                                        conditions.append((condition, result_value, result_value))

                                # Boş alan kontrolü yapalım
                                if all([cond[0].strip() and cond[1].strip() for cond in
                                        conditions]):  # Eğer tüm alanlar doluysa
                                    try:
                                        # Koşullu hesaplamayı loc[] kullanarak uygulayalım
                                        for cond in conditions:
                                            if num_conditions == 1:
                                                # True ve False değerleri ayrı ayrı işleniyor
                                                p2_df.loc[eval(cond[0]), new_label] = cond[1]
                                                p2_df.loc[~eval(cond[0]), new_label] = cond[2]
                                            else:
                                                # Koşullar için tek bir değer işleniyor
                                                p2_df.loc[eval(cond[0]), new_label] = cond[1]

                                        # Güncel DataFrame'i gösterelim
                                        st.write(f"Yeni {new_label} değişkeni türetildi. Güncel DataFrame:")
                                        st.write(p2_df)

                                        # İşlem tamamlandı mesajını columns 2'ye yazdıralım
                                        fecol2.write(f"{new_label} değişkeni başarıyla türetildi.")

                                    except Exception as e:
                                        st.error(f"Hata oluştu: {str(e)}")
                                else:
                                    st.warning("Lütfen koşul, true değer ve false değer alanlarını doldurun.")



                            ##################################################################################
                            ##################   FEATURE EXTRACTION - Nümerik değişkenlerde özellik etkileşimi
                            ##################################################################################

                            if selected_feature == "Nümerik değişkenlerde özellik etkileşimiyle değişken türetmek":
                                # Kullanıcıdan yeni değişken ismi alalım (NEW_ otomatik olarak eklenecek)
                                user_label = fecol1.text_input(f'Türetilecek yeni değişken için bir isim girin (NEW_):',
                                                               key=f"user_label_{counter}")
                                new_label = f"NEW_{user_label}"

                                # Kullanıcıdan matematiksel işlemi girmesini isteyelim
                                user_code = fecol1.text_area(
                                    f"Matematiksel işlem girin (örn. 'p2_df[\"SibSp\"] + p2_df[\"Parch\"] + 1' veya 'p2_df[\"Age\"] ** 2' veya 'p2_df[\"Age\"] * p2_df[\"Survived\"]'):",
                                    key=f"user_code_{counter}"
                                )

                                # Kullanıcının kodu girip girmediğini kontrol edelim
                                if user_code.strip():  # Eğer kod boş değilse
                                    try:
                                        # Tam kodu oluşturalım
                                        full_code = f'p2_df["{new_label}"] = {user_code}'

                                        # Kodun doğru şekilde oluşturulduğunu görmek için bir debug mesajı ekleyelim
                                        st.write(f"Çalıştırılacak kod: {full_code}")

                                        # Kodu çalıştır
                                        exec(full_code)

                                        # Güncel DataFrame'i gösterelim
                                        st.write(f"Yeni {new_label} değişkeni başarıyla türetildi. Güncel DataFrame:")
                                        st.write(p2_df)

                                        # İşlem tamamlandı mesajını columns 2'ye yazdıralım
                                        fecol2.write(f"{new_label} değişkeni başarıyla türetildi.")

                                    except SyntaxError as se:
                                        st.error(f"Sözdizimi hatası: {str(se)}. Girdiğiniz kodu kontrol edin.")
                                    except Exception as e:
                                        st.error(f"Bir hata oluştu: {str(e)}")
                                else:
                                    st.warning("Lütfen geçerli bir matematiksel işlem girin.")




                        # Kullanıcıya başka bir işlem yapmak isteyip istemediği soruluyor
                        continue_or_not = st.radio("Başka bir işlem yapmak istiyor musunuz?", ("Evet", "Hayır"),
                                                   index=1, key=f"radio_{counter}")
                        if continue_or_not == "Hayır":
                            st.write("Döngü sonlandırıldı.")
                            break

                    counter += 1










                ## Kullanıcı bu veri seri bir zaman serisidir der ise burası çalışacak!! kodu rev. et!!
                #overview_tab.subheader("Zaman Serisi Ayrıştırması")
                #ts_decompose(df)
                #overview_tab.subheader("Time Series Overview")
                #ts_decompose(df)



        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")


if __name__ == "__main__":
    main()



