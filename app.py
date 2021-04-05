import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
from collections import Counter
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import dash_bootstrap_components as dbc
import time
import imdb
from dash.exceptions import PreventUpdate

# -----------------------------------Some styling--------------------------------
bc4 = dbc.themes.BOOTSTRAP

app = dash.Dash(__name__, external_stylesheets=[bc4], suppress_callback_exceptions=True,
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}])
server = app.server
app.title = 'Inside IMDb'
tabs_styles = {
    'height': '60px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '10px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#f5c518',
    'color': 'white',
    'padding': '10px'
}

# --------------------------------Data Frames---------------------------------------
df=pd.read_csv('IMDb movies.csv',low_memory=False)
df=df[df['year']!='TV Movie 2019']
df.drop(columns=['metascore'], inplace=True)
df['year']=df['year'].astype(int)
df=df[df['year']>=1970]
df['date_published']=pd.to_datetime(df['date_published'])
df_tab2=df.copy()

df.dropna(inplace=True)
df['budget'] = df['budget'].apply(lambda x: re.sub("[^0-9]", "", x))
df['budget'] = df['budget'].astype(float)
df['usa_gross_income'] = df['usa_gross_income'].apply(lambda x: re.sub("[^0-9]", "", x))
df['usa_gross_income'] = df['usa_gross_income'].astype(float)
df['worlwide_gross_income'] = df['worlwide_gross_income'].apply(lambda x: re.sub("[^0-9]", "", x))
df['worlwide_gross_income'] = df['worlwide_gross_income'].astype(float)

df_tab2.drop(columns=['country','date_published','budget', 'usa_gross_income', 'worlwide_gross_income',
       'reviews_from_users', 'reviews_from_critics','writer','production_company','votes','description'],inplace=True)
df_tab2.dropna(inplace=True)
# dropdown options
#dropdown genres
genres_ = list(df['genre'])
gen_ = []
for i in genres_:
    i = list(i.split(','))
    for j in i:
        gen_.append(j.replace(' ', ""))
g_ = Counter(gen_)
generos_ = list(g_.keys())
options_genero_ = [{'label': i, 'value': i} for i in generos_]
options_genero_.insert(0,{'label': 'All', 'value': 'All'})
#dropdown years
options_year=[{'label':str(i),'value':i} for i in range(1970,2021)]
options_year.reverse()
options_year.insert(0,{'label': 'All', 'value':'All' })

#dropdown ratings
options_ratings=[{'label':str(i)+'+','value':i} for i in range(1,10)]

# dropdown sortby
options_sortby=[{'label':'Newest First','value':'year,False'},
{'label':'Oldest First','value':'year,True'},
{'label':'Top IMDB','value':'avg_vote,False'},
{'label':'Bottom IMDB','value':'avg_vote,True'}]

# ---------------------------------Graphics functions-----------------------------
def normal_round(number):
    """Round a float to the nearest integer."""
    return int(number + 0.5)


def bar_chart_ratings(year):
    aux = df[(df['year'] >= year[0]) & (df['year'] <= year[1])]
    aux['avg_vote_rounded'] = aux.loc[:, 'avg_vote'].apply(lambda x: normal_round(x))
    aux = aux.groupby(by='avg_vote_rounded').size().to_frame().rename(columns={0: 'Count'}).reset_index()
    for i in range(1, 11):
        if i not in aux['avg_vote_rounded'].values:
            aux = aux.append({'avg_vote_rounded': i, 'Count': 0}, ignore_index=True)

    aux = aux.sort_values(by=['avg_vote_rounded'])
    # colors = ['lightslategray',] * 10
    fig = go.Figure(data=[go.Bar(
        x=aux['Count'],
        y=aux['avg_vote_rounded'].apply(lambda x: (str(x) + ' Stars  ') if x != 1 else (str(x) + ' Star  ')),
        # aux_1['avg_vote_rounded'],
        orientation='h',
        marker=dict(
            color='#b1210c'),

        text=aux['Count'], textposition='outside'
        #     marker_color=colors # marker color can be a single color value or an iterable,
        , textfont=dict(family='Helvetica', color='white', size=14)
    )

    ])
    fig.update_layout(  # title_text='Distribution of ratings',
        yaxis=dict(showticklabels=True, type='category', tickfont=dict(family='Helvetica', color='white', size=14)
                   ),
        xaxis=dict(showgrid=False, showline=False, showticklabels=False),
        xaxis_range=[0, 3.5e3],
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0,0)',autosize = True
        #width=730, margin=dict(l=1, r=1, b=100, t=100, pad=4)
    )
    return fig


def donut_chart_ratings_director(year):
    aux = df[(df['year'] >= year[0]) & (df['year'] <= year[1])]
    aux = aux.sort_values(by=['avg_vote'], ascending=False).head(10)[['original_title', 'avg_vote', 'director']]
    aux = aux.reset_index(drop=True)
    fig = px.sunburst(aux, path=['director', 'original_title'], values=[1] * 10, color='avg_vote',
                      color_continuous_scale=px.colors.sequential.YlOrRd)
    fig.layout.coloraxis.colorbar.title = 'Average Vote'
    fig.update_layout(hoverlabel_font=dict(family='Helvetica', size=12), plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)')
    fig.update_traces(textfont=dict(family='Helvetica', size=10), marker_colorbar_tickfont_color="white",
                      hovertemplate='<b>%{label} </b> <br> Director %{parent} <br> Average Vote: %{color:.2f}')
    fig.update_coloraxes(colorbar_tickfont=dict(family='Helvetica', color='white', size=14),
                         colorbar_title_font=dict(family='Helvetica', color='white', size=14))


    return fig


def count_bolinhas(year):
    aux = df[(df['year'] >= year[0]) & (df['year'] <= year[1])]
    aux = aux.groupby(by='year').size().to_frame().rename(columns={0: 'Count'}).reset_index()
    fig = go.Figure(data=[go.Scatter(
        x=aux['year'],
        y=aux['Count'],
        mode='markers',
        marker=dict(colorscale="YlOrRd",
            color=aux['Count'],
            size=aux['Count'] * 0.5,
            showscale=False
        ))])
    fig.layout.template = 'seaborn'

    fig.update_layout(xaxis_title="Year Released", yaxis_title="Total",
                      font=dict(family='Helvetica', color='white', size=14), plot_bgcolor='rgba(0, 0, 0, 0)',
                      paper_bgcolor='rgba(0, 0, 0,0)')

    if (year[1] - year[0]) <= 4:
        fig.update_xaxes(showgrid=False, showline=True, linewidth=1, linecolor='white',
                         tickfont=dict(family='Helvetica', color='white', size=14), tickmode='linear')
    else:
        fig.update_xaxes(showgrid=False, showline=True, linewidth=1, linecolor='white',
                         tickfont=dict(family='Helvetica', color='white', size=14))

    fig.update_yaxes(zeroline=False, showgrid=False, showline=True, linewidth=1, linecolor='white',
                     tickfont=dict(family='Helvetica', color='white', size=14))

    return fig


def line_chart(year):
    aux = df.groupby(by=['year'])[['worlwide_gross_income', 'budget']].sum()
    aux.reset_index(inplace=True)
    aux = aux[(aux['year'] >= year[0]) & (aux['year'] <= year[1])]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=aux['year'], y=aux['worlwide_gross_income'],
                             mode='lines+markers',
                             name='Worlwide Gross Income',
                             marker=dict(symbol='star-dot', size=10,color='#ffc166'
)))
    fig.add_trace(go.Scatter(x=aux['year'], y=aux['budget'],
                             mode='lines+markers',
                             name='Budget', marker=dict(symbol='star-dot', size=10,color='red'
)))
    fig.layout.template = 'seaborn'

    fig.update_layout(xaxis_title="Year", yaxis_title="Amount in dollars",
                      font=dict(family='Helvetica', color='white', size=14), plot_bgcolor='rgba(0, 0, 0, 0)',
                      paper_bgcolor='rgba(0, 0, 0,0)')
    if (year[1] - year[0]) <= 4:
        fig.update_xaxes(showgrid=False, showline=True, linewidth=1, linecolor='white',
                         tickfont=dict(family='Helvetica', color='white', size=14), tickmode='linear')
    else:
        fig.update_xaxes(showgrid=False, showline=True, linewidth=1, linecolor='white',
                         tickfont=dict(family='Helvetica', color='white', size=14))
    fig.update_yaxes(zeroline=False, showgrid=False, showline=True, linewidth=1, linecolor='white',
                     tickfont=dict(family='Helvetica', color='white', size=14)
                     )

    return fig


def table_atores(year):
    aux = df[(df['year'] >= year[0]) & (df['year'] <= year[1])]
    aux = aux['actors'].str.split(',').apply(pd.Series, 1).stack()
    aux.index = aux.index.droplevel(-1)
    aux = aux.to_frame().rename(columns={0: 'Actor'}).groupby(by=['Actor']).size().to_frame().rename(
        columns={0: 'Number of movies'}).sort_values(by='Number of movies', ascending=False)
    aux.reset_index(inplace=True)
    aux = aux.iloc[:10, :]
    return html.Div(
        [
            dash_table.DataTable(
                data=aux.to_dict("rows"),
                columns=[{"id": x, "name": x} for x in aux.columns],

                # table interactivity
                editable=False,
                sort_action="native",
                sort_mode="multi",
                # style table
                style_table={
                    'maxHeight': '60ex',
                    'overflowY': 'scroll',
                    'width': '100%',
                    'minWidth': '100%',

                },
                # style cell
                style_cell={
                    'fontFamily': 'Open Sans',
                    'textAlign': 'center',
                    'height': '60px',
                    'padding': '2px 22px',
                    'whiteSpace': 'inherit',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                    'color': 'white',
                    'backgroundColor': '#1f1f1f'

                },

                # style header
                style_header={
                    'fontWeight': 'bold',
                    'backgroundColor': '#f5c518'  # 'white',
                },

            )
        ]
    )


def indicator(year, genre):
    aux = df[(df['year'] >= year[0]) & (df['year'] <= year[1])]
    genres = list(aux['genre'])
    gen = []
    for i in genres:
        i = list(i.split(','))
        for j in i:
            gen.append(j.replace(' ', ""))
    g = Counter(gen)
    g = {k: v / len(aux) * 100 for k, v in dict(g).items()}
    if genre in g.keys():
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=np.round(g[genre], 1),
            title={'text': genre, "font":{"family":'Helvetica', "color":'white'}},

            domain={'x': [0, 1], 'y': [0, 1]},
            number={'prefix': "%", "font": {"family": 'Helvetica', "color": 'white'}}, gauge={
                'axis': {'range': [None, 100], "tickcolor": "white",
                         "tickfont": {"family": 'Helvetica', "color": 'white'}},
                "bar": {"color": "#b1210c", "line": {"color": "white"}}, "bordercolor": "white"}))

    else:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=0,
            title={'text': genre, "font": {"family": 'Helvetica', "color": 'white'}},

            domain={'x': [0, 1], 'y': [0, 1]},
            number={'prefix': "%", "font": {"family": 'Helvetica', "color": 'white'}}, gauge={
                'axis': {'range': [None, 100], "tickcolor": "white",
                         "tickfont": {"family": 'Helvetica', "color": 'white'}},
                "bar": {"color": "#b1210c", "line": {"color": "white"}}, "bordercolor": "white"}))

    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0,0)', autosize=True)
    return fig


def movies_tab2(genres, years, rating, sortby):
    df_movies = df_tab2.copy()
    #     filter by genre:
    aux = pd.DataFrame()
    if type(genres) == str:
        if genres == 'All':
            aux=df_movies.copy()
        else:
            aux = aux.append(df_movies[df_movies['genre'].str.contains(genres)])
    if type(genres) == list:
        if 'All' in genres:
            aux=df_movies.copy()
        else:
            for genre in genres:
                aux = aux.append(df_movies[df_movies['genre'].str.contains(genre)])

    aux.drop_duplicates(inplace=True)
    #     filter by year:
    if type(years)==list:
        if 'All' in years:
            aux=aux.copy()
        else:
            aux = aux[aux['year'].isin(years)]
    if type(years)==int:
        aux = aux[aux['year']==years]
    if years == 'All':
        aux = aux.copy()
    #     filter by rating
    aux = aux[aux['avg_vote'] >= rating]
    #     filtersort by

    sortby = sortby.split(',')
    sortby_by = sortby[0]
    if sortby[1] == 'True':
        sortby_asc = True
    else:
        sortby_asc = False
    aux.sort_values(by=sortby_by, ascending=sortby_asc, inplace=True)
    return aux.head()
# ----------------------------------The app itself---------------------------------
app.layout = html.Div([
    html.Div([
        html.Header([
html.Div([
            html.Div([html.Img(id="logo-image",
                              className="logo",
                              src=app.get_asset_url('icon imdb.png'),
                              alt="logo"
                               ),
                                dcc.Markdown(
                                   id="source",
                                   children=[
                                       "Source: [Inside Kaggle](https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset?select=IMDb+movies.csv)"
                                   ],style={ 'font-size': '12px'}
                               ),

                      ],style={'display':'inline-block','width':'10%'}),
            #html.Div([
                html.Div([
                    html.Div(children=[html.H6("This application illustrates the movies presented by the IMDb's website from 1970 until 2020, with all the information associated with it as actors, directors and ratings.",
                                          ),
                html.H6( 'This dashboard is composed of two tabs, being them fully interactive.')])],className='header_intro'),#'white-space': 'nowrap','display': 'inline-block'
                ]),
                html.Div([
                    html.Br(),
                    html.H6( "Authorship: Catarina Candeias, Catarina Urbano, Rita Ferreira, Rebeca Pinheiro" )
                    ],className='header_footer')
                    ]
                     )], className='header')
             #                 ])
    ,
    html.Div([
        dcc.Tabs(
            id='tabs', value='1', children=[
                dcc.Tab(label='Discover IMDb', value='1', style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label='Search Movies', value='2', style=tab_style, selected_style=tab_selected_style)
            ], style=tabs_styles
        ),
        html.Div(id='tab-output')
    ])
])
@app.callback(Output('tab-output', 'children'), [Input('tabs', 'value')])
def display_content(value):
    if value == '1':
        return html.Div([
            html.Div([
                html.H6('Choose Year(s)'),
                dcc.RangeSlider(
                    id='year_slider',
                    min=1970,
                    max=2020,
                    value=[1970, 2020],
                    marks={str(i): str(i) for i in range(1970, 2021,2)}, step=1, className='slider')
            ], className='box'),
            # .........................1st visual.............

            html.Div([
                dbc.Row(
                    [
                        dbc.Col(html.Div([
                            html.Div(
                                children=[html.H2('User Rating'),
                                          html.H6(
                                              'Displays the number of IMDb registered users that voted for each rating option')]),
                            html.Div(dcc.Loading(id="loading-1",
                                                 children=[html.Div(dcc.Graph(id='bar-charts')),
                                                           html.Div([html.Div(id="loading-output-1")]),], type="circle", color='#f5c518')),

                        ], style={'height': '600px'},className='box')),

                        # .........................2nd visual...............
                        dbc.Col(
                            html.Div([
                                html.Div(
                                    children=[html.H2('Popularity by film genre'),
                                              html.H6("Exhibits the percentage of the movie's genres")]
                                ),
                                html.Div(
                                    id="div-dropdown-1",
                                    className="div-for-dropdown",
                                    children=[html.Br(),
                                        # Dropdown for locations on map
                                        dcc.Dropdown(
                                            className='teste',
                                            id='dcc_genre_dropdown',
                                            options=[{'label': i, 'value': i} for i in generos_],
                                            placeholder="Select Genre",
                                            style={'max-width': '250px','color':'white','background-color':'#262525'}
                                        ),
                                        # html.Div(
                                        #    dcc.Graph(id='indicator'))
                                    ]
                                ),
                                html.Div(dcc.Loading(id="loading-2",
                                                     children=[html.Div(dcc.Graph(id='indicator', style={
                                                         'height': 400,
                                                         'width': 600,
                                                         "display": "block",
                                                         "margin-left": "auto",
                                                         "margin-right": "auto",
                                                     })),html.Div([html.Div(id="loading-output-2")])], type="circle",color='#f5c518'))

                            ],style={'height': '600px'}, className='box')
                        )
                    ])]),
            # .........................3rd visual...............
            html.Div([
                dbc.Row(
                    [
                        dbc.Col(html.Div([
                            html.Div(
                                children=[html.H2('Actors with the most movies made'),
                                          html.H6(
                                              "Shows the 10 actors with more movies’ appearances, as well as its total number")]),
                            html.Div(dcc.Loading(id="loading-3",
                                                 children=[html.Div(id="update-table"),html.Div([html.Div(id="loading-output-3")])],
                                                 type="circle",color='#f5c518'))
                        ],style={'height': '600px'}, className='box')),

                        # .........................4th visual...............
                        dbc.Col(html.Div([
                            html.Div(
                                children=[html.H2('Movies and Directors'),
                                          html.H6(
                                              'Presents the 10 highest-rated movies and their respective director'),
                                          html.Br()]
                            ),

                            html.Div(dcc.Loading(id="loading-4",
                                                 children=[html.Div(dcc.Graph(id='donut-chart')),html.Div([html.Div(id="loading-output-4")])],
                                                 type="circle",color='#f5c518'))

                        ],style={'height': '600px'}, className='box'))
                    ]
                )
            ]),
            # .........................5th visual...............
            html.Div([
                            html.Div([
                                html.Div(
                                    children=[html.H2('Number of Movies released by year'),
                                              html.H6(
                                                  'Slide the cursor over the circle to know the number of movies released in each year')]
                                ),

                                html.Div(dcc.Loading(id="loading-5",
                                                     children=[html.Div(dcc.Graph(id='bolinhas-chart')),html.Div([html.Div(id="loading-output-5")])],
                                                     type="circle",color='#f5c518'))
                            ], className='box')
        ]),
                        # .........................6th visual...............
            html.Div([
                            html.Div([
                                html.Div(
                                    children=[html.H2('Budget vs Gross Income for worldwide movies'),
                                              html.Br()
                                              ]
                                ),

                                html.Div(dcc.Loading(id="loading-6",
                                                     children=[html.Div(dcc.Graph(id='line-chart')),html.Div([html.Div(id="loading-output-6"),])],
                                                     type="circle",color='#f5c518'))

                            ], className='box')
            ])
                    ])

    elif value == '2':
        return html.Div([
                    html.Div([html.Div([
                        html.Div(
                            id="div-dropdown-1",
                            className="div-for-dropdown1",
                            children=[
                                html.Div(html.Label('Genre',style={'color':'white'}),style={'text-align':'center','margin-right': '15%'}),#50px
                                dcc.Dropdown(
                                    id='genre_dropdown',
                                    options=options_genero_,
                                    placeholder="Select Genre(s)",
                                    multi=True,
                                    style={'max-width': '300px','background-color':'#262525'},
                                    value='All')

                            ],style={'width': '19%', 'display': 'inline-block'}
                        ),
                        html.Div(
                            id="div-dropdown-2",
                            className="div-for-dropdown2",
                            children=[
                                html.Div(html.Label('Year', style={'color': 'white'}),
                                         style={'text-align': 'center', 'margin-right': '15%'}),
                                dcc.Dropdown(
                                     id='year_dropdown',
                                    options=options_year,
                                    placeholder="Select Year(s)",
                                    multi=True,
                                    style={'max-width': '300px','background-color':'#262525'},
                                    value='All'

                                ),
                            ],style={'width': '19%', 'display': 'inline-block'}
                        ),
                        html.Div(
                            className="div-for-dropdown3",
                            children=[
                                html.Div(html.Label('Rating', style={'color': 'white'}),
                                         style={'text-align': 'center', 'margin-right': '15%'}),
                                dcc.Dropdown(
                                     id='ratings_dropdown',
                                    className='teste',
                                    options=options_ratings,
                                    placeholder="Select Rating",
                                    multi=False,
                                    style={'max-width': '300px','color':'white','background-color':'#262525'},
                                    value=1,
                                    clearable=False
                                ),
                            ],style={'width': '19%', 'display': 'inline-block'}
                        ),
                        html.Div(
                            id="div-dropdown-4",
                            className='teste',
                            children=[
                                html.Div(html.Label('Sort', style={'color': 'white'}),
                                         style={'text-align': 'center', 'margin-right': '15%'}),
                                dcc.Dropdown(className='teste',
                                     id='sortby_dropdown',
                                    options=options_sortby,
                                    multi=False,
                                    style={'max-width': '300px','color':'white','background-color':'#262525'},
                                    value='year,False',
                                    clearable=False
                                )
                            ],style={'width': '19%', 'display': 'inline-block'}
                        ),
                            html.Button('Submit', id='my-button', style={'width': '19%', 'display': 'inline-block','height':'45px','margin-block-start': 'auto'}),

],style={'border-radius': '5px','background-color':'#1f1f1f','margin': '10px','padding': '15px',
         'box-shadow': '2px 2px 2px lightgrey','display':'flex'}),
                        html.Div(html.Label('Please make sure that at least one option is selected in each filter',
                                   style={'color':'black','font-weight': 'bold'}),style={'text-align':'center'})
                    ]),
                        html.Div([dcc.Loading(color='#f5c518',children=[
                            html.Div([
                                html.Div(html.H4(id='titulo1'),style={'font-weight': 'bold','color':'white'}),
                                html.Br(),
                                html.Div(html.Img(id='imagem1', style={'width': '60%','margin-left': 'auto','margin-right': 'auto',
                                                                       'text-align':'center'}),style={'text-align':'center'})],className='col1'),#,
                            html.Div([
                                html.Br(),
                                html.Div(html.H6(id='description1')),
                                html.Br(),
                                html.Div(html.H6(id='genre1')),
                                html.Div(html.H6(id='rating1')),
                                html.Div(html.H6(id='duration1')),
                                html.Div(html.H6(id='language1'))
                            ],className='col2')], type="circle")
                        ],className='box2'),
                        html.Div([
                            dcc.Loading(color='#f5c518', children=[html.Div([
                                html.Div(html.H4(id='titulo2'), style={'font-weight': 'bold','color':'white'}),
                                html.Br(),
                                html.Div(html.Img(id='imagem2', style={'width': '60%','margin-left': 'auto','margin-right': 'auto',
                                                                       'text-align':'center'}),style={'text-align':'center'})],className='col1'),

                            html.Div([
                                html.Br(),
                                html.Div(html.H6(id='description2')),
                                html.Br(),
                                html.Div(html.H6(id='genre2')),
                                html.Div(html.H6(id='rating2')),
                                html.Div(html.H6(id='duration2')),
                                html.Div(html.H6(id='language2'))
                            ],className='col2')], type="circle")

                        ],className='box2'),
                        html.Div([
                            dcc.Loading(color='#f5c518',children=[html.Div([
                                html.Div(html.H4(id='titulo3'), style={'font-weight': 'bold','color':'white'}),
                                html.Br(),
                                html.Div(html.Img(id='imagem3', style={'width': '60%','margin-left': 'auto','margin-right': 'auto',
                                                                       'text-align':'center'}),style={'text-align':'center'})],className='col1'),
                            html.Div([
                                html.Br(),
                                html.Div(html.H6(id='description3')),
                                html.Br(),
                                html.Div(html.H6(id='genre3')),
                                html.Div(html.H6(id='rating3')),
                                html.Div(html.H6(id='duration3')),
                                html.Div(html.H6(id='language3'))
                                    ],className='col2')], type="circle")
                        ],className='box2'),
                        html.Div([
                            dcc.Loading(color='#f5c518',children=[html.Div([
                                html.Div(html.H4(id='titulo4'), style={'font-weight': 'bold','color':'white'}),
                                html.Br(),
                                html.Div(html.Img(id='imagem4', style={'width': '60%','margin-left': 'auto','margin-right': 'auto',
                                                                       'text-align':'center'}),style={'text-align':'center'})],className='col1'),
                            html.Div([
                                html.Br(),
                                html.Div(html.H6(id='description4')),
                                html.Br(),
                                html.Div(html.H6(id='genre4')),
                                html.Div(html.H6(id='rating4')),
                                html.Div(html.H6(id='duration4')),
                                html.Div(html.H6(id='language4'))
                                  ],className='col2')], type="circle")
                        ],className='box2'),
                        html.Div([
                            dcc.Loading(color='#f5c518',children=[html.Div([
                                html.Div(html.H4(id='titulo5'), style={'font-weight': 'bold','color':'white'}),
                                html.Br(),
                                html.Div(html.Img(id='imagem5', style={'width': '60%','margin-left': 'auto','margin-right': 'auto',
                                                                       'text-align':'center'}),style={'text-align':'center'})],className='col1'),
                            html.Div([
                                html.Br(),
                                html.Div(html.H6(id='description5')),
                                html.Br(),
                                html.Div(html.H6(id='genre5')),
                                html.Div(html.H6(id='rating5')),
                                html.Div(html.H6(id='duration5')),
                                html.Div(html.H6(id='language5'))
                                    ],className='col2')], type="circle")
                        ],className='box2')

                ])

@app.callback(
[Output('titulo1','children'),
Output('imagem1','src'),
Output('description1','children'),
Output('genre1','children'),
Output('rating1','children'),
Output('duration1','children'),
Output('language1','children'),

Output('titulo2','children'),
Output('imagem2','src'),
Output('description2','children'),
Output('genre2','children'),
Output('rating2','children'),
Output('duration2','children'),
Output('language2','children'),

Output('titulo3','children'),
Output('imagem3','src'),
Output('description3','children'),
Output('genre3','children'),
Output('rating3','children'),
Output('duration3','children'),
Output('language3','children'),

Output('titulo4','children'),
Output('imagem4','src'),
Output('description4','children'),
Output('genre4','children'),
Output('rating4','children'),
Output('duration4','children'),
Output('language4','children'),

Output('titulo5','children'),
Output('imagem5','src'),
Output('description5','children'),
Output('genre5','children'),
Output('rating5','children'),
Output('duration5','children'),
Output('language5','children')
 ],
[Input('my-button', 'n_clicks')],
state=[State('genre_dropdown', 'value'),
       State('year_dropdown', 'value'),
       State('ratings_dropdown', 'value'),
       State('sortby_dropdown', 'value')])

def compute(n_clicks, genres, years, rating,sortby):
    movies_filtered = movies_tab2(genres, years, rating,sortby)
    movies_filtered['imdb_title_id'] = movies_filtered['imdb_title_id'].apply(lambda x: x[2:])
    movies_filtered['duration'] = movies_filtered['duration'].apply(lambda x: 'Duration: ' + str(x) + ' minutes')
    movies_filtered['language'] = movies_filtered['language'].apply(lambda x: 'Language(s): ' + str(x))

    titulos = []
    url_imgs = []
    descriptions = []
    genres_list=[]
    ratings = []
    durations = []
    languages = []
    access = imdb.IMDb()
    for i, movie_id in enumerate(movies_filtered['imdb_title_id'].values):

        movie = access.get_movie(str(movie_id))

        titulos.append(("%s (%s)" % (movie['title'], movie['year'])))
        url_imgs.append(movie['cover url'])
        if 'plot outline' in movie.data.keys():
            descriptions.append(movie['plot outline'])
        elif 'plot' in movie.data.keys():
            descriptions.append(movie['plot'][0].split('::')[0])
        else:
            descriptions.append('Description: Not available')
        genres_list.append('Genre(s): '+ str(movies_filtered['genre'].values[i]))
        ratings.append('Rating: '+ str(movies_filtered['avg_vote'].values[i]))
        durations.append(movies_filtered['duration'].values[i])
        languages.append(movies_filtered['language'].values[i])

    return (titulos[0],url_imgs[0],descriptions[0],genres_list[0],ratings[0],durations[0],languages[0],
    titulos[1], url_imgs[1], descriptions[1], genres_list[1], ratings[1], durations[1], languages[1],
    titulos[2], url_imgs[2], descriptions[2], genres_list[2], ratings[2], durations[2], languages[2],
    titulos[3], url_imgs[3], descriptions[3], genres_list[3], ratings[3], durations[3], languages[3],
    titulos[4], url_imgs[4], descriptions[4], genres_list[4], ratings[4], durations[4], languages[4])




@app.callback(Output('bar-charts', "figure"), Input('year_slider', 'value'))
def input_triggers_spinner(year):
    time.sleep(1)
    return bar_chart_ratings(year)

@app.callback(Output('indicator', 'figure'), [Input('year_slider', 'value'),Input('dcc_genre_dropdown', 'value')])
def input_triggers_spinner(year,genre):
    time.sleep(1)
    return indicator(year, genre)

@app.callback(Output('update-table', 'children'), Input('year_slider', 'value'))
def input_triggers_spinner(year):
    time.sleep(1)
    return table_atores(year)

@app.callback(Output('donut-chart', 'figure'), Input('year_slider', 'value'))
def input_triggers_spinner(year):
    time.sleep(1)
    return donut_chart_ratings_director(year)

@app.callback(Output('bolinhas-chart', 'figure'), Input('year_slider', 'value'))
def input_triggers_spinner(year):
    time.sleep(1)
    return count_bolinhas(year)

@app.callback(Output('line-chart', 'figure'), Input('year_slider', 'value'))
def input_triggers_spinner(year):
    time.sleep(1)
    return line_chart(year)

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)