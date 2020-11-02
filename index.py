# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 20:12:07 2020

@author: 13154
"""
##mutiple-page    
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import apyori as ap
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px

# Since we're adding callbacks to elements that don't exist in the app.layout,
# Dash will raise an exception to warn us that we might be
# doing something wrong.
# In this case, we're adding the elements through a callback, so we can ignore
# the exception.
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

pd.read_csv('employee_attrition.csv')

employee_attrition = pd.read_csv('employee_attrition.csv')
employee_attrition = employee_attrition.dropna()
employee_attrition = employee_attrition.drop_duplicates()
employee_attrition.isnull().any().any()
employee_attrition = employee_attrition.drop(['EmployeeCount','StandardHours'],axis = 1)
employee_attrition = employee_attrition[employee_attrition['Age'] - employee_attrition['YearsAtCompany'] >= 0]
employee_attrition = employee_attrition[employee_attrition['Age'] - employee_attrition['YearsInCurrentRole'] >= 0]
employee_attrition = employee_attrition[employee_attrition['Age'] - employee_attrition['YearsSinceLastPromotion'] >= 0]
employee_attrition = employee_attrition[employee_attrition['Age'] - employee_attrition['YearsWithCurrManager'] >= 0]

employee_attrition["Age"] = pd.qcut(employee_attrition.Age, 3,
                                   labels = ['low_age','med_age','high_age'])
employee_attrition["DailyRate"] = pd.qcut(employee_attrition.DailyRate, 4,
                                   labels = ['DailyRate = bad','DailyRate = fair','DailyRate = good','DailyRate = excellent'])
employee_attrition["DistanceFromHome"] = pd.qcut(employee_attrition.DistanceFromHome, 3,
                                   labels = ['DistanceFromHome = far','DistanceFromHome = fair','DistanceFromHome = near'])
employee_attrition["EmployeeNumber"] = pd.qcut(employee_attrition.EmployeeNumber, 4,
                                   labels = ['less','med','more','much more'])
employee_attrition["HourlyRate"] = pd.qcut(employee_attrition.HourlyRate, 4,
                                   labels = ['HourlyRate = bad','HourlyRate = fair','HourlyRate = good','HourlyRate = excellent'])
employee_attrition["MonthlyIncome"] = pd.qcut(employee_attrition.MonthlyIncome, 3,
                                   labels = ['MonthlyIncome = low','MonthlyIncome = med','MonthlyIncome = high'])
employee_attrition["MonthlyRate"] = pd.qcut(employee_attrition.MonthlyRate, 4,
                                   labels = ['MonthlyRate = bad','MonthlyRate = fair','MonthlyRate = good','MonthlyRate = excellent'])
employee_attrition["NumCompaniesWorked"] = pd.qcut(employee_attrition.NumCompaniesWorked, 3,
                                   labels = ['NumCompaniesWorked = less','NumCompaniesWorked = med','NumCompaniesWorked = more'])
employee_attrition["PercentSalaryHike"] = pd.qcut(employee_attrition.PercentSalaryHike, 3,
                                   labels = ['PercentSalaryHike = less','PercentSalaryHike = med','PercentSalaryHike = more'])
employee_attrition["TotalWorkingYears"] = pd.qcut(employee_attrition.TotalWorkingYears, 4,
                                   labels = ['TotalWorkingYears = less','TotalWorkingYears = med','TotalWorkingYears = more'
                                             ,'TotalWorkingYears = much more'])
employee_attrition["YearsAtCompany"] = pd.qcut(employee_attrition.YearsAtCompany, 3,
                                   labels = ['YearsAtCompany = less','YearsAtCompany = med','YearsAtCompany = more'])
employee_attrition["YearsInCurrentRole"] = pd.qcut(employee_attrition.YearsInCurrentRole, 3,
                                   labels = ['YearsInCurrentRole = less','YearsInCurrentRole = med','YearsInCurrentRole = more'])
employee_attrition["YearsWithCurrManager"] = pd.qcut(employee_attrition.YearsWithCurrManager, 3,
                                   labels = ['YearsWithCurrManager = less',
                                             'YearsWithCurrManager = med','YearsWithCurrManager = ore'])
employee_attrition["TrainingTimesLastYear"] = pd.qcut(employee_attrition.TrainingTimesLastYear, 3,
                                   labels = ['TrainingTimesLastYear = less','TrainingTimesLastYear = med'
                                             ,'TrainingTimesLastYear = more'])
employee_attrition["YearsSinceLastPromotion"] = pd.cut(employee_attrition.YearsSinceLastPromotion, [-1,6,10,30],
                                   labels = ['YearsSinceLastPromotion = less','YearsSinceLastPromotion = med',
                                             'YearsSinceLastPromotion = more'])
employee_attrition["Education"] = pd.cut(employee_attrition.Education, 5,
                                   labels = ['Education = low_edu','Education = basic_edu',
                                             'Education = med_edu','Education = high_edu','Education = great_edu'])
employee_attrition["EnvironmentSatisfaction"] = pd.cut(employee_attrition.EnvironmentSatisfaction, 4,
                                   labels = ['EnvironmentSatisfaction = bad','EnvironmentSatisfaction = fair',
                                             'EnvironmentSatisfaction = good','EnvironmentSatisfaction = excellent'])
employee_attrition["JobInvolvement"] = pd.cut(employee_attrition.JobInvolvement, 4,
                                   labels = ['JobInvolvement = bad','JobInvolvement = fair',
                                             'JobInvolvement = good','JobInvolvement = excellent'])
employee_attrition["JobLevel"] = pd.cut(employee_attrition.JobLevel, 5,
                                   labels = ['JobLevel = level1','JobLevel = level2','JobLevel = level3'
                                             ,'JobLevel = level4','JobLevel = level5'])
employee_attrition["JobSatisfaction"] = pd.cut(employee_attrition.JobSatisfaction, 4,
                                   labels = ['JobSatisfaction = bad','JobSatisfaction = fair',
                                             'JobSatisfaction = good','JobSatisfaction = excellent'])
employee_attrition["PerformanceRating"] = pd.cut(employee_attrition.PerformanceRating, 2,
                                   labels = ['PerformanceRating = fair','PerformanceRating = good'])
employee_attrition["RelationshipSatisfaction"] = pd.cut(employee_attrition.RelationshipSatisfaction, 4,
                                   labels = ['RelationshipSatisfaction = bad','RelationshipSatisfaction = fair',
                                             'RelationshipSatisfaction = good','RelationshipSatisfaction = excellent'])
employee_attrition["StockOptionLevel"] = pd.cut(employee_attrition.StockOptionLevel, 4,
                                   labels = ['StockOptionLevel = level1','StockOptionLevel = level2',
                                             'StockOptionLevel = level3','StockOptionLevel = level4'])
employee_attrition["WorkLifeBalance"] = pd.cut(employee_attrition.WorkLifeBalance, 4,
                                   labels = ['WorkLifeBalance = bad','WorkLifeBalance = fair',
                                             'WorkLifeBalance = good','WorkLifeBalance = excellent'])
####transforma data into transaction to fit the model##########
records= []
for i in range(0,len(employee_attrition)):
    records.append([str(employee_attrition.values[i,j]) 
    for j in range(0, len(employee_attrition.columns))])
frequent_itemset = ap.apriori(records, min_support=0.8, min_confidence=0.8,
                              min_lift=1,min_length=2)
results = list(frequent_itemset)
len(results)
results[1:5]
te = TransactionEncoder()
te_ary = te.fit(records).transform(records)
df_tran = pd.DataFrame(te_ary, columns=te.columns_)
available_indicators = employee_attrition.columns.unique()
####transforma data into transaction to fit the model##########
####data preparation for slider##########
min_sup = np.array([0.1,0.2,0.3,.4,.5,.6,.7,.8,.9])
min_cof = np.array([0.1,0.2,0.3,.4,.5,.6,.7,.8,.9])
min_lif = np.array([.6,.8,1.0,1.2,1.4])
consequent = ['No','Yes']
##############
########descriptive indicators###########
available_indicators = employee_attrition.columns.unique()

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


index_page = html.Div([
    dcc.Link('Descriptive Exploratory', href='/page-1'),
    html.Br(),
    dcc.Link('Visulization of ARA', href='/page-2'),
])

#############################page1 layout########################################
page_1_layout = html.Div([
    html.H1('Descriptive Exploratory'),
    html.H5("Change the value to see distribution"),  
##bar chart part
    dcc.Dropdown(
                id='column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='Age'
            ),
   dcc.Graph(id='indicator-graphic'),
##bar chart part
    
    dcc.Link('Go to Page 2', href='/page-2'),
    html.Br(),
    dcc.Link('Go back to home', href='/')
   ]
)

@app.callback(
    Output('indicator-graphic', 'figure'),
    [Input('column', 'value')])

def update_graph(column):
    df = employee_attrition.groupby([column, 'Attrition']).size().reset_index(name='Counts') 
    fig = px.bar(df, x=column, y='Counts', color="Attrition")
    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
    fig.update_xaxes(title=column) 
    return fig
#############################page2 layout########################################
page_2_layout = html.Div([
    html.H1('Results of ARA'),
    html.H1("Hyperparameter tuning"),
    dcc.Graph(id='indicator-graphic2'),
    html.H5("Change mininum support"),
    dcc.Slider(
        id='min_sup-slider',
        min=min_sup.min(),
        max=min_sup.max(),
        value=min_sup.mean(),
        marks={str(sup): str(sup) for sup in min_sup},
        step=None
    ),
    html.H5("Change mininum confidence"),
    dcc.Slider(
        id='min_cof-slider',
        min=min_cof.min(),
        max=min_cof.max(),
        value=min_cof.mean(),
        marks={str(cof): str(cof) for cof in min_cof},
        step=None
    ),
    html.H5("Change mininum lift"),
    dcc.Slider(
        id='min_lif-slider',
        min=min_lif.min(),
        max=min_lif.max(),
        value=min_lif.mean(),
        marks={str(lif): str(lif) for lif in min_lif},
        step=None
    ),

    html.Div(id='page-2-content'),
    html.Br(),
    dcc.Link('Go to Page 1', href='/page-1'),
    html.Br(),
    dcc.Link('Go back to home', href='/')
])

@app.callback(
    Output('indicator-graphic2', 'figure'),
    [Input('min_sup-slider', 'value'),
     Input('min_cof-slider', 'value'),
     Input('min_lif-slider', 'value')])
def ara_results(min_sup,min_conf,min_lift):
    consequent = ['No','Yes']
    frequent_itemsets = apriori(df_tran, min_sup, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_conf)
    rules = rules[rules['lift'] > min_lift]
    sup_rules = pd.DataFrame()
    for i in consequent:
        df = rules[rules['consequents'] == {i}]
        sup_rules = sup_rules.append(df,ignore_index = True)
    fig = px.scatter(sup_rules,x='support',y='confidence',
                 color = 'lift',size = 'lift')
    return(fig)


# Update the index
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout
    elif pathname == '/page-2':
        return page_2_layout
    else:
        return index_page
    # You could also return a 404 "URL not found" page here


if __name__ == '__main__':
    app.run_server(debug=True)


