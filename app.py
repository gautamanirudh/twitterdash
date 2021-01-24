import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash_table import DataTable
from dash_table.FormatTemplate import Format
from plotly.subplots import make_subplots
import pandas as pd
# import advertools as adv
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import twint
from datetime import date
from datetime import timedelta
import numpy as np
# nest_asyncio.apply()
import time
import math
import plotly.graph_objects as go
import string
import dash_extensions as de  # pip install dash-extension
# ml
import pickle
import re
import nltk
from nltk.corpus import stopwords
stop = stopwords.words("english")
from nltk.stem.porter import PorterStemmer
english_stemmer=nltk.stem.SnowballStemmer('english')






lottie1_url = "https://assets10.lottiefiles.com/datafiles/9jPPC5ogUyD6oQq/data.json"
options = dict(loop=True, autoplay=True, rendererSettings=dict(preserveAspectRatio='xMidYMid slice'))
def get_yesterday_date():
    return date.today()-timedelta(day=1)

searchedQuery = ""


twtr_lang_df = pd.read_csv('twitter_languages.csv')
lang_options = [{'label': loc_name, 'value': code}
                for loc_name, code in
                zip(twtr_lang_df['name'] + ' | ' + twtr_lang_df['local_name'],
                    twtr_lang_df['code'])]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY],   meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}])

server = app.server

x = np.arange(10)

defaultfig = go.Figure(data=go.Scatter(x=x, y=x**2))

# navbar

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

nav_item = dbc.NavItem(dbc.NavLink("Github Repository", href="https://github.com/gautamanirudh/twitterdash/tree/master", target="_blank", style={"color":"white"}))

dropdown = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem("Anirudh Gautam", href="https://github.com/gautamanirudh" ,target="_blank"),
        dbc.DropdownMenuItem(divider=True),
        dbc.DropdownMenuItem("Kawaljeet Singh Batra", href="https://github.com/Kawaljeet2001", target="_blank"),
        dbc.DropdownMenuItem(divider=True),
        dbc.DropdownMenuItem("Harshit Dave", href="https://github.com/harshitkd", target="_blank"),
    ],
    style={"color":"white"},
    nav=True,
    in_navbar=True,
    label="Connect",
)


logo = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                        dbc.Col(dbc.NavbarBrand("Real Time Sentiment Analysis", className="ml-2")),
                    ],
                    align="center",
                    no_gutters=True,
                ),
            ),
                dbc.NavbarToggler(id="navbar-toggler2"),
            dbc.Collapse(
                dbc.Nav(
                    [nav_item,dropdown], className="ml-auto", navbar=True
                ),
                id="navbar-collapse2",
                navbar=True,
            ),
        ]
    ),
    color="#00acee",
    dark=True,
    className="mb-5",
)
# headimage

image = html.Div([
        html.Img(src="https://media.giphy.com/media/l378c04F2fjeZ7vH2/giphy.gif", style={"width":"32%", "border-radius":4, "height":"15vw","align":"left","padding-right":5}),
        html.Img(src="https://media.giphy.com/media/xUPGcEOEllmvFAvako/giphy.gif", style={"width":"32%", "border-radius":4, "height":"15vw","align":"center"}),
        html.Img(src="https://media.giphy.com/media/TJP7EH5i1fB2rKeWbf/giphy.gif", style={"width":"32%", "border-radius":4, "height":"15vw","align":"right","padding-left":5}),
],
style={"widht":"5rem", "text-align":"center"},)


app.layout = html.Div([

    logo,

dbc.Row(
    [
        dbc.Col(
            [
                   image
            ]
        )
       
    ],className="border border-primary"),


html.Div(id="query_output",className="alert alert-warning d-none",role="alert"),
dbc.Row(
    [

        dbc.Col(
            [
                html.P("Select Service",className="badge badge-primary"),
                dcc.Dropdown(
        id='dropdown',
        options=[
            {'label': 'Keyword', 'value': 'keyword'},
            {'label': 'Username', 'value': 'username'},
            
        ],
        value='username'
    ),
            ],width="4",className="no-gutters"
        ),
        
        dbc.Col(
            
            [
                html.P("Select Service",className="badge badge-primary"),
                dbc.Input(id='twitter_search',
            placeholder='Search query'),
            ],width="4"
        ),
    #     dbc.Col(
    #         [
    #              dcc.DatePickerRange(
    #     id='date_picker',
    #     min_date_allowed=date(1995, 8, 5),
    #     max_date_allowed=date.today(),
    #     initial_visible_month=str(date.today()),
    #     end_date=str(date.today()),
    # ),
    #         ],width="3"
    #     ),
        dbc.Col(
           
            [
                html.P("Tweets limit",className="badge badge-primary"),
        dcc.Slider(
        id='my_slider',
        min=1000,
        max=10000,
        step=1000,
        value=1000,
        marks={i:str(i) for i in range(1000,10001,1000)}
    ),
            ],width="4"
        ),
        
    
    
   
   ],className="pl-5 pr-5"
),
dbc.Row(
    dbc.Col(
[
    dbc.Button(id='search_button', children='Submit', outline=True),
],width=12
    ),className="pl-5 pr-5"
),

dbc.Row([
    
    dbc.Spinner(color="primary", type="grow",fullscreen=True,children=[
    html.Div([
        dbc.Row([
            dbc.Col([
                html.P("Total Tweets Fetched"),
                html.P(id="out_total_tweets")

            ],className="border border-info rounded"),

            dbc.Col([
                html.P("Average Likes"),
                html.P(id="out_avg_likes")

            ],className="border border-info rounded"),

            dbc.Col([
                html.P("Average Retweets"),
                html.P(id="out_avg_retweets")

            ],className="border border-info rounded"),

            dbc.Col([
                html.P("Average Replies"),
                html.P(id="out_avg_replies")

            ],className="border border-info rounded"),

            dbc.Col([
                html.P("Average Engagement"),
                html.P(id="out_avg_engagement")

            ],className="border border-info rounded"),
        ]),
        dbc.Row(

            [
                dbc.Col([
                    dbc.Row(
                        [
                            dbc.Col(de.Lottie(options=options, width="50%", height="50%", url=lottie1_url),),
                            dbc.Col( [html.H3(id="perc_tweets_hash"),
                                        html.P("of tweets contains hashtags") 
                                        ])
                    
                        ]
                    )
                ],width=4),

                  dbc.Col([
                    dbc.Row(
                        [
                            dbc.Col(de.Lottie(options=options, width="50%", height="50%", url=lottie1_url),),
                            dbc.Col( [html.H3(id="perc_tweets_img"), html.P("of tweets contains images") ])
                    
                        ]
                    )
                ],width=4),

                  dbc.Col([
                    dbc.Row(
                        [
                            dbc.Col(de.Lottie(options=options, width="50%", height="50%", url=lottie1_url),),
                            dbc.Col( [html.H3(id="perc_tweets_videos"), html.P("of tweets contains videos") ])
                    
                        ]
                    )
                ],width=4)
                
            ]
        ),
        
    dbc.Row(
        [
            dbc.Col([dcc.Loading(dcc.Graph(id='fig1', ),)],width="auto"),
             dbc.Col([dcc.Loading(dcc.Graph(id='fig2', ))],width="auto"),
        ]
    ),
    dbc.Row(
        [
            dbc.Col([dcc.Loading(dcc.Graph(id='fig3', ))],width="auto"),
            dbc.Col([dcc.Loading(dcc.Graph(id='fig4', ))],width="auto"),
        ]
    ),
    dbc.Row(
        [
             dbc.Col([dcc.Loading(dcc.Graph(id='fig5', ))],width="auto"),
              dbc.Col([dcc.Loading(dcc.Graph(id='fig6', ))],width="auto"),
        ]
    ),
    dbc.Row(
        [
            dbc.Col([dcc.Loading(dcc.Graph(id='fig7', ))],width="auto"),
            dbc.Col([dcc.Loading(dcc.Graph(id='fig8', ))],width="auto"),
        ]
    ),
],id="charts_output",style={"display":"none"}),
    ]),

],className="pl-5 pr-5")

])
# config={'displayModeBar': False},


@app.callback(
    [Output("charts_output","style"),
    Output("query_output","children"),
    Output("query_output","className"),
    Output("out_total_tweets","children"),
    Output("out_avg_likes","children"),
    Output("out_avg_retweets","children"),
    Output("out_avg_replies","children"),
    Output("out_avg_engagement","children"),
    Output("perc_tweets_hash","children"),
    Output("perc_tweets_img","children"),
    Output("perc_tweets_videos","children"),
    Output("fig1","figure"),Output("fig2","figure"),
    Output("fig3","figure"),Output("fig4","figure"),
    Output("fig5","figure"),Output("fig6","figure"),
    Output("fig7","figure"),Output("fig8","figure")],
    [Input("search_button","n_clicks")],
    [State("dropdown","value"),State("twitter_search","value"),State("my_slider","value")]
)
def outputfun(n_clicks,dropdown,queryval,limit_value):
    def sentiment_predict(df):
        

        def cleaning(tweet , remove_stopwords = True):
            text = re.sub("[^a-zA-Z]"," ", tweet)
            words =text.lower().split()

            if remove_stopwords:
                stops = set(stopwords.words("english"))
                words = [w for w in words if not w in stops]

            b=[]
            stemmer = english_stemmer 
            for word in words:
                b.append(stemmer.stem(word))


            return(b)


        clean_Text = []
        for review in df['tweet']:
            clean_Text.append( " ".join(cleaning(review)))

        vectorizer = pickle.load(open("final_vectorizer.pkl" , "rb"))
        kmeans = pickle.load(open("final_kmeans.pkl" , "rb"))
        transformed_text = vectorizer.transform(clean_Text)
        clusters = kmeans.predict(transformed_text)
        unique, frequency = np.unique(clusters,return_counts = True)
        output_dic = {}
        
        output_dic['positive'] = frequency[1]
        output_dic['neutral'] = frequency[0]
        output_dic['negative'] = frequency[2]
        
        return output_dic



    if dropdown=="username":
        if n_clicks is None:
            raise PreventUpdate

        if queryval:
            searchedQuery = queryval
            # print("start",type(start_date))
            # print("end",str(end_date))
            # print("limit",str(limit_value))
            #**************************************************************************************************
                    
            s = time.time()
            t = twint.Config()
            # if start_date:
            #     t.Since = str(start_date)
            #     t.Until = str(end_date),
            username = searchedQuery
            # print("start",type(start_date))
            # print("end",type(end_date))
            t.Search = f"from:@{username}"

            t.Store_object = True

            t.Limit = limit_value 
            t.Pandas = True
            
            twint.run.Search(t)
            
            e = time.time()

            print(e-s)

            tweetsdata = twint.storage.panda.Tweets_df #Scraped Dataframe

            if tweetsdata.shape[0]==0:
                return (dash.no_update,["No Data Available"],"alert alert-warning",dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update)

            #Filtering
            tweetsdata = tweetsdata[["date","timezone","place","tweet","language","hashtags","day","hour","link","urls","photos","video","nlikes","nreplies","nretweets"]]
            tweetsdata["date"] = pd.to_datetime(tweetsdata["date"])
            tweetsdata.rename(columns={'date':"datetime"},inplace=True)
            tweetsdata["date"] = tweetsdata["datetime"].dt.date
            tweetsdata["engagement"] = tweetsdata["nreplies"] + tweetsdata["nretweets"]

            #Basic Features
            #**********************************************************************************************************
            total_tweets = tweetsdata.shape[0]
            avg_likes = math.ceil(tweetsdata["nlikes"].mean())
            avg_replies =  math.ceil(tweetsdata["nreplies"].mean())
            avg_retweets =  math.ceil(tweetsdata["nretweets"].mean())
            avg_engagement = tweetsdata.loc[:,"engagement"].mean()

           



            #Tweets with Hashtags
            tweetsdata["ishash"] = tweetsdata["hashtags"].apply(lambda x: 1 if len(x)>0 else 0)
            tweets_with_hash = tweetsdata[tweetsdata["ishash"]==1].shape[0]
            tweets_with_0hash = total_tweets-tweets_with_hash
            
            tweets_with_hash_perc = round(tweets_with_hash/total_tweets,2)*100
            tweets_with_hash_perc = f'{tweets_with_hash_perc}%'
            #Tweets with Videos
            tweets_with_videos = tweetsdata[tweetsdata["video"]==1].shape[0]
            tweets_with_videos_perc = round(tweets_with_videos/total_tweets,2)*100
            tweets_with_videos_perc = f'{tweets_with_videos_perc}%'

            #Tweets with Image
            tweetsdata["isimage"] = tweetsdata["photos"].apply(lambda x: 1 if len(x)>0 else 0)
            tweets_with_images = tweetsdata[tweetsdata["isimage"]==1].shape[0]
            tweets_with_images_perc = round((tweets_with_images/total_tweets),2)*100
            tweets_with_images_perc = f'{tweets_with_images_perc}%'


            # Figure 1 Sentiment analysis

             #SENTIMENT PREDICTION
            sentiment_prediction_dic = sentiment_predict(tweetsdata)

            fig1 = go.Figure(data=[go.Pie(labels=list(sentiment_prediction_dic.keys()), values=list(sentiment_prediction_dic.values()),hole=0.3,textinfo='label+percent',
                                        )],
                        layout = dict(
                        font = dict(
            #                                     family =  'Raleway',
                                                size =  16,
            #                                     color = '#7f7f7f'
                                    )))
            fig1.update_layout(title_text='Sentiments of Tweets', title_x=0.5,title_y=0.1)




            #Figure 2 Tweets vs date

        
            fig2 = go.Figure(
            data = [go.Scatter(
            x=tweetsdata.groupby(by="date").count().index, y=tweetsdata.groupby(by="date").count()["datetime"],

            )],
                
                                layout={
                                    
                                    "font" : {
                                                #  family =  'Raleway',
                                                "size" :  16,
                                                #  "color" : '#7f7f7f'
                                        },
                                    
                                    'title':{
                                        'text':"Tweets vs Days",
                                        'y':0.9,
                                            'x':0.5,
                                        'xanchor':'center',
                                        'yanchor':'top'
                                    },
                                    'yaxis':{
                                        
                                        "title":"Number of Tweets"
                                    },
                                    'xaxis':{
                                        "title":"Date",
            #                               "tickvals" : list(range(24)),
                                    }
                                }
                                )
        


            #Language analysis

            twtr_lang_df = pd.read_csv('twitter_languages.csv')
            twtr_lang_df = twtr_lang_df[["code","name"]]


            tweets_lang_count = tweetsdata.groupby(by="language").count()

            if "und" in tweets_lang_count.index:
                tweets_lang_count.drop("und",axis=0,inplace=True)
            tweets_lang_count = tweets_lang_count.sort_values(by="datetime",ascending=False)["datetime"]

            n_lang_to_include = 0
            tweets_lang_count_dic = {}
            for i in range(min(6,tweets_lang_count.shape[0])): 
                
                n_lang_to_include = i
                if tweets_lang_count.index[i] not in list(twtr_lang_df["code"]):
            #         print(twtr_lang_df["code"])
                
                    break


            if n_lang_to_include==0:
                tweets_lang_count_dic = tweets_lang_count[:4]
                tweets_lang_count_dic["others"] = tweets_lang_count[4:].sum()
            else:
                tweets_lang_count_dic = tweets_lang_count[:n_lang_to_include]
                tweets_lang_count_dic["Others"] = tweets_lang_count[n_lang_to_include:].sum()
                

            def get_lang(lang_code):
                try:
                    return list(twtr_lang_df.query("code==@lang_code")["name"])[0]
                except:
                    return lang_code

            tweets_lang_count_dic = {get_lang(k) : v for k,v in tweets_lang_count_dic.items()}

# 
            #Figure 3 Language analysis Pie chart

            fig3 = go.Figure(data=[go.Pie(labels=list(tweets_lang_count_dic.keys()), values=list(tweets_lang_count_dic.values()),hole=0.3,textinfo='label+percent',
                                        )],
                        layout = dict(
                        font = dict(
            #                                     family =  'Raleway',
                                                size =  16,
            #                                     color = '#7f7f7f'
                                    )))
            fig3.update_layout(title_text='Language Distribution of Tweets', title_x=0.5,title_y=0.1)


            #Weekdays analysis

            #1 - Monday
            weekday_tweets = tweetsdata.groupby("day").count()["datetime"]
            weekday_tweets_dict = {k:v for k,v in zip(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],list(weekday_tweets))}



            fig4 = go.Figure(data=[go.Bar(
            x=list(weekday_tweets_dict.values())[::-1],y=list(weekday_tweets_dict.keys())[::-1],
            width=0.6 ,# customize width here,
            orientation="h",
            text = list(weekday_tweets_dict.values())[::-1],
            textposition = "inside",

            marker=dict(
                color='rgba(50, 171, 90, 0.6)',
                line=dict(
                    color='rgba(50, 171, 90, 1.0)',
                    width=1),
            ),

            )],
                                
                                layout = dict(
                                    yaxis = dict(
                                        showgrid=False,
                                        showline=False,
                                        showticklabels=True,
                                        

                                    ),
                                    
                                    xaxis = dict(
                                        zeroline=False,
                                        showline=False,
                                        showticklabels=True,
                                        showgrid=True,
            #                                 domain=[0, 0.42],
                                    ),
                                    font = dict(
            #                                     family =  'Raleway',
                                            size =  16,
                                            color = '#7f7f7f'
                                    )
                                )
                                )
            fig4.update_layout(title_text='Distribution of Tweets ()', title_x=0.5)


            #Teets vs Hours analysis
            hour_tweets_dict = {k:v for k,v in zip(range(24),tweetsdata.groupby("hour").count()["datetime"])}

            fig5 = go.Figure(
            data = [go.Scatter(
                x=list(hour_tweets_dict.keys()),
                y=list(hour_tweets_dict.values()),

            )],
                
                                layout={
                                    "font" : {
                                                #  family =  'Raleway',
                                                "size" :  16,
                                                #  "color" : '#7f7f7f'
                                        },
                                    'title':{
                                        'text':"Tweets VS Hours",
                                        'y':0.9,
                                            'x':0.5,
                                        'xanchor':'center',
                                        'yanchor':'top'
                                    },
                                    'yaxis':{
                                        
                                        "title":"Number of Tweets"
                                    },
                                    'xaxis':{
                                        "title":"Hours   (12 am - 11 pm)",
                                        "dtick":True,
                                        "tickvals" : list(range(24))
                                    }
                                }
                                )

            #Hashtags analysis


            tweetsdata_hashtags = (tweetsdata.explode("hashtags").reset_index())
            tweetsdata_hashtags = tweetsdata_hashtags.replace({'hashtags':{'nan':np.nan}}).dropna(subset=['hashtags'])
            tweetsdata_hashtags_grp_freg = tweetsdata_hashtags.groupby("hashtags").count().sort_values("datetime").tail(10)["datetime"]



            fig6 = go.Figure(data=[go.Bar(
            x=tweetsdata_hashtags_grp_freg,y=tweetsdata_hashtags_grp_freg.index,
                width=0.6 ,# customize width here,
                orientation="h",
                text = tweetsdata_hashtags_grp_freg,
                textposition = "outside",
                textangle=0,
            
            
                marker=dict(
                    color='rgba(50, 171, 90, 0.6)',
                    line=dict(
                        color='rgba(50, 171, 90, 1.0)',
                        width=1),
                ),
                
            )],
                                
                                layout = dict(
                                    yaxis = dict(
                                            showgrid=False,
                                            showline=False,
                                            showticklabels=True,
                                            
                                    

                                    ),
                                    
                                    xaxis = dict(
                                            zeroline=False,
                                            showline=False,
                                            showticklabels=True,
                                            showgrid=True,
            #                                 domain=[0, 0.42],
                                        ),
                                    font = dict(
            #                                     family =  'Raleway',
                                                size =  16,
                                                color = '#7f7f7f'
                                    )
                                )
                                )
            fig6.update_layout(title_text='Frequency of Hashtags Used', title_x=0.5)

            #Figure 7

            tweetsdata_hashtags_grp_engag = tweetsdata_hashtags.groupby("hashtags").sum().sort_values(by="engagement").tail(10)["engagement"]


            fig7 = go.Figure(data=[go.Bar(
            x=tweetsdata_hashtags_grp_engag,y=tweetsdata_hashtags_grp_engag.index,
                width=0.6 ,# customize width here,
                orientation="h",
                text = tweetsdata_hashtags_grp_engag,
                textposition = "auto",
                textangle=0,
                
            
            
                marker=dict(
                    color='rgba(50, 171, 90, 0.6)',
                    line=dict(
                        color='rgba(50, 171, 90, 1.0)',
                        width=1),
                ),
                
            )],
                                
                                layout = dict(
                                    yaxis = dict(
                                            showgrid=False,
                                            showline=False,
                                            showticklabels=True,
                                            
                                    

                                    ),
                                    
                                    xaxis = dict(
                                            zeroline=False,
            #                                 visible = False,
                                            showline=False,
            #                                 showticklabels=True,
                                            showgrid=True,
            #                                 domain=[0, 0.42],
                                        ),
                                    font = dict(
            #                                     family =  'Raleway',
                                                size =  16,
                                                color = '#7f7f7f'
                                    )
                                )
                                )
            fig7.update_layout(title_text='Hashtags with Most engagement (Retweets + Replies)', title_x=0.5)


            #Figure 8

            tweetsdata_hashtags_grp_likes = tweetsdata_hashtags.groupby("hashtags").sum().sort_values(by="nlikes").tail(10)["nlikes"]


            fig8 = go.Figure(data=[go.Bar(
            x=tweetsdata_hashtags_grp_likes,y=tweetsdata_hashtags_grp_likes.index,
                width=0.6 ,# customize width here,
                orientation="h",
                text = tweetsdata_hashtags_grp_likes,
                textposition = "auto",
                textangle=0,
                
            
            
                marker=dict(
                    color='rgba(50, 171, 90, 0.6)',
                    line=dict(
                        color='rgba(50, 171, 90, 1.0)',
                        width=1),
                ),
                
            )],
                                
                                layout = dict(
                                    yaxis = dict(
                                            showgrid=False,
                                            showline=False,
                                            showticklabels=True,
                                            
                                    

                                    ),
                                    
                                    xaxis = dict(
                                            zeroline=False,
            #                                 visible = False,
                                            showline=False,
            #                                 showticklabels=True,
                                            showgrid=True,
            #                                 domain=[0, 0.42],
                                        ),
                                    font = dict(
            #                                     family =  'Raleway',
                                                size =  16,
                                                color = '#7f7f7f'
                                    )
                                )
                                )
            fig8.update_layout(title_text='Hashtags with Most number of likes', title_x=0.5)
            #**************************************************************************************************


            print(queryval)
            return ({'display': 'block'},[],
            "alert alert-warning d-none",
            total_tweets,avg_likes,
            avg_retweets,avg_retweets,
            avg_engagement,
            tweets_with_hash_perc,
            tweets_with_images_perc,
            tweets_with_videos_perc,
            fig1,
            fig2,
            fig3,
            fig4,
            fig5,
            fig6,
            fig7,
            fig8
             )
            # except Exception as e:
            #     print(e)
            #     return (dash.no_update,["No user exists"],dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update)
        else:
            return (dash.no_update,["Enter Username"],"alert alert-warning",dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update)

    else:

        if queryval:

            # n_clicks,dropdown,queryval,start_date,end_date,limit_value
            c = twint.Config()

            c.Search = queryval
            c.Limit = limit_value
            c.Pandas = True

            twint.run.Search(c)

            
            querydata = twint.storage.panda.Tweets_df

            if querydata.shape[0]==0:
                return (dash.no_update,["No Data Available"],"alert alert-warning",dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update)


            querydata = querydata[["date","timezone","place","tweet","language","hashtags","day","hour","link","urls","photos","video","nlikes","nreplies","nretweets"]]
            querydata["engagement"] = querydata["nreplies"]+querydata["nretweets"]

            querydata["date"] = pd.to_datetime(querydata["date"])
            querydata.rename(columns={'date':"datetime"},inplace=True)
            querydata["date"] = querydata["datetime"].dt.date


            # querydata.drop("datetime",axis=1,inplace=True)


            
            def count_words(s):
                res = sum([i.strip(string.punctuation).isalpha() for i in s.split()]) 
                return res
            querydata["nwords_per_tweet"] = querydata['tweet'].apply(count_words)
            querydata["nwords_per_tweet"].mean()
            
            # import math
            total_tweets = querydata.shape[0]
            avg_likes = math.ceil(querydata["nlikes"].mean())
            avg_replies =  math.ceil(querydata["nreplies"].mean())
            avg_retweets =  math.ceil(querydata["nretweets"].mean())
            avg_engagement = math.ceil(querydata["engagement"].mean())
            

            #Tweets with Hashtags
            querydata["ishash"] = querydata["hashtags"].apply(lambda x: 1 if len(x)>0 else 0)
            tweets_with_hash = querydata[querydata["ishash"]==1].shape[0]
            tweets_with_0hash = total_tweets-tweets_with_hash
            
            tweets_with_hash_perc = round(tweets_with_hash/total_tweets,2)*100
            tweets_with_hash_perc = f'{tweets_with_hash_perc}%'

            #Tweets with Videos
            tweets_with_videos = querydata[querydata["video"]==1].shape[0]
            tweets_with_videos_perc = round(tweets_with_videos/total_tweets,2)*100
            tweets_with_videos_perc = f'{tweets_with_videos_perc}%'

            #Tweets with Image
            querydata["isimage"] = querydata["photos"].apply(lambda x: 1 if len(x)>0 else 0)
            tweets_with_images = querydata[querydata["isimage"]==1].shape[0]
            tweets_with_images_perc = round((tweets_with_images/total_tweets),2)*100
            tweets_with_images_perc = f'{tweets_with_images_perc}%'       
            #querydata_exploded_hashtags


            #ENGAGED TWEETS
            engaged_tweetsdata = querydata.loc[querydata["engagement"]>avg_engagement]

            engaged_tweetsdata_exploded = engaged_tweetsdata.explode("hashtags")
            top10_hash = engaged_tweetsdata_exploded.dropna().groupby("hashtags").count().sort_values("tweet",ascending=False).head(10)["tweet"]

            #Language analysis

            tweets_lang_count = querydata.groupby(by="language").count().drop("und",axis=0).sort_values(by="tweet",ascending=False)["tweet"]
            n_lang_to_include = 0
            tweets_lang_count_dic = {}
            twtr_lang_df = pd.read_csv('twitter_languages.csv')
            twtr_lang_df = twtr_lang_df[["code","name"]]
            for i in range(6): 
                
                n_lang_to_include = i
                if tweets_lang_count.index[i] not in list(twtr_lang_df["code"]):
            #         print(twtr_lang_df["code"])
                
                    break

            if n_lang_to_include==0:
                tweets_lang_count_dic = tweets_lang_count[:4]
                tweets_lang_count_dic["others"] = tweets_lang_count[4:].sum()
            else:
                tweets_lang_count_dic = tweets_lang_count[:n_lang_to_include]
                tweets_lang_count_dic["Others"] = tweets_lang_count[n_lang_to_include:].sum()
                



            def get_lang(lang_code):
                try:
                    return list(twtr_lang_df.query("code==@lang_code")["name"])[0]
                except:
                    return lang_code

            tweets_lang_count_dic = {get_lang(k) : v for k,v in tweets_lang_count_dic.items()}


            # Figure 1 Sentiment analysis

             #SENTIMENT PREDICTION
            sentiment_prediction_dic = sentiment_predict(querydata)

            fig1 = go.Figure(data=[go.Pie(labels=list(sentiment_prediction_dic.keys()), values=list(sentiment_prediction_dic.values()),hole=0.3,textinfo='label+percent',
                                        )],
                        layout = dict(
                        font = dict(
            #                                     family =  'Raleway',
                                                size =  16,
            #                                     color = '#7f7f7f'
                                    )))
            fig1.update_layout(title_text='Sentiments of Tweets', title_x=0.5,title_y=0.1)


                #Figure 2 Tweets vs date

        
            fig2 = go.Figure(
            data = [go.Scatter(
            x=querydata.groupby(by="date").count().index, y=querydata.groupby(by="date").count()["datetime"],

            )],
                
                layout={
                    
                    "font" : {
                                #  family =  'Raleway',
                                "size" :  16,
                                #  "color" : '#7f7f7f'
                        },
                    
                    'title':{
                        'text':"Tweets vs Days",
                        'y':0.9,
                            'x':0.5,
                        'xanchor':'center',
                        'yanchor':'top'
                    },
                    'yaxis':{
                        
                        "title":"Number of Tweets"
                    },
                    'xaxis':{
                        "title":"Date",
    #                               "tickvals" : list(range(24)),
                    }
                }
                )
            





            
                #Figure 2 Language analysis Pie chart

            fig3 = go.Figure(data=[go.Pie(labels=list(tweets_lang_count_dic.keys()), values=list(tweets_lang_count_dic.values()),hole=0.3,textinfo='label+percent',
                                        )],
                        layout = dict(
                        font = dict(
            #                                     family =  'Raleway',
                                                size =  16,
            #                                     color = '#7f7f7f'
                                    )))
            fig3.update_layout(title_text='Language Distribution of Tweets', title_x=0.5,title_y=0.1)


            querydata_hashtags = (querydata.explode("hashtags").reset_index())
            querydata_hashtags = querydata_hashtags.replace({'hashtags':{'nan':np.nan}}).dropna(subset=['hashtags'])
            querydata_hashtags_grp_freg = querydata_hashtags.groupby("hashtags").count().sort_values("datetime").tail(10)["datetime"]


    #Weekdays analysis

            #1 - Monday
            weekday_tweets = querydata.groupby("day").count()["datetime"]
            weekday_tweets_dict = {k:v for k,v in zip(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],list(weekday_tweets))}



            fig4 = go.Figure(data=[go.Bar(
            x=list(weekday_tweets_dict.values())[::-1],y=list(weekday_tweets_dict.keys())[::-1],
            width=0.6 ,# customize width here,
            orientation="h",
            text = list(weekday_tweets_dict.values())[::-1],
            textposition = "inside",

            marker=dict(
                color='rgba(50, 171, 90, 0.6)',
                line=dict(
                    color='rgba(50, 171, 90, 1.0)',
                    width=1),
            ),

            )],
                                
                                layout = dict(
                                    yaxis = dict(
                                        showgrid=False,
                                        showline=False,
                                        showticklabels=True,
                                        

                                    ),
                                    
                                    xaxis = dict(
                                        zeroline=False,
                                        showline=False,
                                        showticklabels=True,
                                        showgrid=True,
            #                                 domain=[0, 0.42],
                                    ),
                                    font = dict(
            #                                     family =  'Raleway',
                                            size =  16,
                                            color = '#7f7f7f'
                                    )
                                )
                                )
            fig4.update_layout(title_text='Distribution of Tweets ()', title_x=0.5)


            #Teets vs Hours analysis
            hour_tweets_dict = {k:v for k,v in zip(range(24),querydata.groupby("hour").count()["datetime"])}

            fig5 = go.Figure(
            data = [go.Scatter(
                x=list(hour_tweets_dict.keys()),
                y=list(hour_tweets_dict.values()),

            )],
                
                                layout={
                                    "font" : {
                                                #  family =  'Raleway',
                                                "size" :  16,
                                                #  "color" : '#7f7f7f'
                                        },
                                    'title':{
                                        'text':"Tweets VS Hours",
                                        'y':0.9,
                                            'x':0.5,
                                        'xanchor':'center',
                                        'yanchor':'top'
                                    },
                                    'yaxis':{
                                        
                                        "title":"Number of Tweets"
                                    },
                                    'xaxis':{
                                        "title":"Hours   (12 am - 11 pm)",
                                        "dtick":True,
                                        "tickvals" : list(range(24))
                                    }
                                }
                                )







            fig6 = go.Figure(data=[go.Bar(
            x=querydata_hashtags_grp_freg,y=querydata_hashtags_grp_freg.index,
                width=0.6 ,# customize width here,
                orientation="h",
                text = querydata_hashtags_grp_freg,
                textposition = "outside",
                textangle=0,
            
            
                marker=dict(
                    color='rgba(50, 171, 90, 0.6)',
                    line=dict(
                        color='rgba(50, 171, 90, 1.0)',
                        width=1),
                ),
                
            )],
                                
                                layout = dict(
                                    yaxis = dict(
                                            showgrid=False,
                                            showline=False,
                                            showticklabels=True,
                                            
                                    

                                    ),
                                    
                                    xaxis = dict(
                                            zeroline=False,
                                            showline=False,
                                            showticklabels=True,
                                            showgrid=True,
            #                                 domain=[0, 0.42],
                                        ),
                                    font = dict(
            #                                     family =  'Raleway',
                                                size =  16,
                                                color = '#7f7f7f'
                                    )
                                )
                                )
            fig6.update_layout(title_text='Frequency of Hashtags Used', title_x=0.5)

            # querydata_hashtags_grp_engag = querydata_hashtags.groupby("hashtags").sum().sort_values(by="engagement").tail(10)["engagement"]




            # Top 10 Hashtags in highly engaged tweets
            fig7 = go.Figure(data=[go.Bar(
                x=top10_hash[::-1],y=top10_hash.index[::-1],
                    width=0.6 ,# customize width here,
                    orientation="h",
                    text = top10_hash[::-1],
                    textposition = "outside",
                    textangle=0,
                
                
                    marker=dict(
                        color='rgba(50, 171, 90, 0.6)',
                        line=dict(
                            color='rgba(50, 171, 90, 1.0)',
                            width=1),
                    ),
                    
                )],
                                    
                                    layout = dict(
                                        yaxis = dict(
                                                showgrid=False,
                                                showline=False,
                                                showticklabels=True,
                                                
                                        

                                        ),
                                        
                                        xaxis = dict(
                                                zeroline=False,
                                                showline=False,
                                                showticklabels=True,
                                                showgrid=True,
                #                                 domain=[0, 0.42],
                                            ),
                                        font = dict(
                #                                     family =  'Raleway',
                                                    size =  16,
                                                    color = '#7f7f7f'
                                        )
                                    )
                                    )
            fig7.update_layout(title_text='Top 10 Hashtags in highly engaged tweets', title_x=0.5)

            #Hashtags with Most number of likes
            tweetsdata_hashtags_grp_likes = querydata_hashtags.groupby("hashtags").sum().sort_values(by="nlikes").tail(10)["nlikes"]


            fig8 = go.Figure(data=[go.Bar(
            x=tweetsdata_hashtags_grp_likes,y=tweetsdata_hashtags_grp_likes.index,
                width=0.6 ,# customize width here,
                orientation="h",
                text = tweetsdata_hashtags_grp_likes,
                textposition = "auto",
                textangle=0,
                
            
            
                marker=dict(
                    color='rgba(50, 171, 90, 0.6)',
                    line=dict(
                        color='rgba(50, 171, 90, 1.0)',
                        width=1),
                ),
                
            )],
                                
                                layout = dict(
                                    yaxis = dict(
                                            showgrid=False,
                                            showline=False,
                                            showticklabels=True,
                                    ),
                                    
                                    xaxis = dict(
                                            zeroline=False,
            #                                 visible = False,
                                            showline=False,
            #                                 showticklabels=True,
                                            showgrid=True,
            #                                 domain=[0, 0.42],
                                        ),
                                    font = dict(
            #                                     family =  'Raleway',
                                                size =  16,
                                                color = '#7f7f7f'
                                    )
                                )
                                )
            fig8.update_layout(title_text='Hashtags with Most number of likes', title_x=0.5)


            return ({'display': 'block'},[],
                    "alert alert-warning d-none",
                    total_tweets,avg_likes,
                    avg_retweets,avg_retweets,
                    avg_engagement,
                    tweets_with_hash_perc,
                    tweets_with_images_perc,
                    tweets_with_videos_perc,
                    fig1,
                    fig2,
                    fig3,
                    fig4,
                    fig5,
                    fig6,
                    fig7,
                    fig8 )
        else:
            return (dash.no_update,["Enter Query"],"alert alert-warning",dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update)

if __name__ == "__main__":
    app.run_server()
