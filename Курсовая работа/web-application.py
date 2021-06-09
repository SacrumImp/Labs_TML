import streamlit as st

from io import StringIO
import graphviz 
import pydotplus
from sklearn.tree import export_graphviz

from sklearn.datasets import *
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score 
from sklearn.model_selection import LeaveOneOut

from sklearn.svm import SVC, NuSVC, LinearSVC, OneClassSVM, SVR, NuSVR, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from io import StringIO
import graphviz 
import pydotplus
from sklearn.tree import export_graphviz

class MetricLogger:
    
    def __init__(self):
        self.df = pd.DataFrame(
            {'metric': pd.Series([], dtype='str'),
            'alg': pd.Series([], dtype='str'),
            'value': pd.Series([], dtype='float')})

    def add(self, metric, alg, value):
        """
        Добавление значения
        """
        # Удаление значения если оно уже было ранее добавлено
        self.df.drop(self.df[(self.df['metric']==metric)&(self.df['alg']==alg)].index, inplace = True)
        # Добавление нового значения
        temp = [{'metric':metric, 'alg':alg, 'value':value}]
        self.df = self.df.append(temp, ignore_index=True)

    def get_data_for_metric(self, metric, ascending=True):
        """
        Формирование данных с фильтром по метрике
        """
        temp_data = self.df[self.df['metric']==metric]
        temp_data_2 = temp_data.sort_values(by='value', ascending=ascending)
        return temp_data_2['alg'].values, temp_data_2['value'].values
    
    def plot(self, str_header, metric, ascending=True, figsize=(5, 5)):
        """
        Вывод графика
        """
        array_labels, array_metric = self.get_data_for_metric(metric, ascending)
        fig, ax1 = plt.subplots(figsize=figsize)
        pos = np.arange(len(array_metric))
        rects = ax1.barh(pos, array_metric,
                         align='center',
                         height=0.5, 
                         tick_label=array_labels)
        ax1.set_title(str_header)
        for a,b in zip(pos, array_metric):
            plt.text(0.5, a-0.05, str(round(b,3)), color='white')
        st.pyplot(fig) 

@st.cache
def getData():
    voices = pd.read_csv('/Users/vladislavalpeev/Documents/labs/Labs_TML/Курсовая работа/voice.csv', sep=",")
    voices = voices.drop_duplicates()
    le = LabelEncoder()
    voices_enc_le = le.fit_transform(voices['label'])
    voices_Y = voices_enc_le
    np.unique(voices_enc_le)
    voices_cor = voices
    voices_cor["label"] = voices_Y
    return voices_cor, voices.corr()

@st.cache
def processData(voices, task_clas_cols):
    return train_test_split(voices[task_clas_cols].values, voices['label'].values, test_size=0.3, random_state=1)

def get_png_tree(tree_model_param, feature_names_param):
    dot_data = StringIO()
    export_graphviz(tree_model_param, out_file=dot_data, feature_names=feature_names_param,
                    filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return graph.create_png()

def clas_train_model(model_name, model, clasMetricLogger):
    model.fit(clas_voices_X_train, clas_voices_Y_train)

    if (model_name == "Tree"):
        st.image(get_png_tree(model, task_clas_cols))

    Y_pred = model.predict(clas_voices_X_test)
    
    precision = precision_score(clas_voices_Y_test, Y_pred)
    recall = recall_score(clas_voices_Y_test, Y_pred)
    f1 = f1_score(clas_voices_Y_test, Y_pred)
    
    clasMetricLogger.add('precision', model_name, precision)
    clasMetricLogger.add('recall', model_name, recall)
    clasMetricLogger.add('f1', model_name, f1)

    fig, ax = plt.subplots(figsize=(10,5))    
    plot_confusion_matrix(model, clas_voices_X_test, clas_voices_Y_test, ax=ax,
                      display_labels=['0','1'], 
                      cmap=plt.cm.Blues, normalize='true')
    fig.suptitle(model_name)
    st.pyplot(fig)

# Начало структуры страницы

st.header('Курсовая работа')

'''
    В качестве набора данных мы будем использовать набор данных для определения пола по голосу - https://www.kaggle.com/primaryobjects/voicegender
    Это задача является интересной, потому как позволяет исследовать уникальные качества голосов людей разных полов.
'''

voices, corMatr = getData()
st.write(voices.head())

st.subheader('Анализ датасета')

variants = voices.columns

first_param = st.selectbox('Выберите параметр:', variants)
second_param = st.selectbox('Выберите второй параметр:', variants.drop(first_param))
st.pyplot(sns.pairplot(voices.drop(variants.drop([first_param, second_param]), 1)))

showCorMatr = st.checkbox('Показать корреляционную матрицу:')
if (showCorMatr):
    fig, ax = plt.subplots(figsize=(17, 12))
    ax.set_title('Корреляционный анализ')
    sns.heatmap(corMatr, annot=True, fmt='.2f')
    st.pyplot(fig)

st.subheader('Решение задачи классификации')

task_clas_cols = ['meanfun', 'IQR', 'Q25', 'sd', 'sp.ent']

clas_voices_X_train, clas_voices_X_test, clas_voices_Y_train, clas_voices_Y_test = processData(voices, task_clas_cols)

variants_models = ['LogR', 'KNN', 'SVC', 'Tree', 'RF', 'GB']

machine_name = st.selectbox('Выберите модель:', variants_models)

# Параметры для моделей машинного обучения:
if (machine_name == 'LogR'):
    machine = LogisticRegression()
if (machine_name == 'KNN'):
    n_neighbors = st.slider('Количество соседей:', min_value=1, max_value=800, value=1, step=2)
    machine = KNeighborsClassifier(n_neighbors=n_neighbors)
if (machine_name == 'SVC'):
    c_param = st.slider('Параметр С:', min_value=1, max_value=1000, value=1, step=10)
    machine = SVC(C = c_param, probability=True)
if (machine_name == 'Tree'):
    n_levels = st.slider('Глубина дерева:', min_value=1, max_value=35, value=1, step=1)
    machine = DecisionTreeClassifier(max_depth=n_levels)
if (machine_name == 'RF'):
    machine = RandomForestClassifier()
if (machine_name == 'GB'):
    machine = GradientBoostingClassifier()

clasMetricLogger = MetricLogger()

clas_train_model(machine_name, machine, clasMetricLogger)