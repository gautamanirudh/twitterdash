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



app.layout = html.Div([
 dcc.Dropdown(
        id='dropdown',
        options=[
            {'label': 'Keyword', 'value': 'keyword'},
            {'label': 'Username', 'value': 'username'},
            
        ],
        value='username'
    ),
    dbc.Input(id='twitter_search',
            placeholder='Search query'),
    dbc.Button(id='search_button', children='Submit', outline=True),
    dcc.DatePickerRange(
        id='date_picker',
        min_date_allowed=date(1995, 8, 5),
        max_date_allowed=date.today(),
        initial_visible_month=str(date.today()),
        end_date=str(date.today()),
    ),
    dcc.Slider(
        id='my_slider',
        min=1000,
        max=10000,
        step=1000,
        value=500,
        marks={i:i for i in range(1000,10001,1000)}
    ),
    html.H4(id="query_output"),
    dbc.Spinner(color="primary", type="grow",fullscreen=True,children=[
    html.Div(id="charts_output",children=[
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
        ]
    ),
],style={"display":"none"}),
    ]),

    
])
# config={'displayModeBar': False},

@app.callback(
    [Output("charts_output","style"),Output("query_output","children"),Output("fig1","figure"),Output("fig2","figure"),Output("fig3","figure"),Output("fig4","figure"),Output("fig5","figure"),Output("fig6","figure"),Output("fig7","figure")],
    [Input("search_button","n_clicks")],
    [State("dropdown","value"),State("twitter_search","value"),State("date_picker","start_date"),State("date_picker","end_date"),State("my_slider","value")]
)
def outputfun(n_clicks,dropdown,queryval,start_date,end_date,limit_value):
    if dropdown=="username":
        if n_clicks is None:
            raise PreventUpdate

        if queryval:
            searchedQuery = queryval
            print("start",type(start_date))
            print("end",str(end_date))
            print("limit",str(limit_value))
            #**************************************************************************************************
                    
            s = time.time()
            t = twint.Config()
            if start_date:
                t.Since = str(start_date)
                t.Until = str(end_date),
            username = searchedQuery
            print("start",type(start_date))
            print("end",type(end_date))
            t.Search = f"from:@{username}"

            t.Store_object = True

            t.Limit = limit_value 
            t.Pandas = True
            
            twint.run.Search(t)
            
            e = time.time()

            print(e-s)

            tweetsdata = twint.storage.panda.Tweets_df #Scraped Dataframe



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

            #Tweets with Hashtags
            tweetsdata["ishash"] = tweetsdata["hashtags"].apply(lambda x: 1 if len(x)>0 else 0)
            tweets_with_hash = tweetsdata[tweetsdata["ishash"]==1].shape[0]
            tweets_with_0hash = total_tweets-tweets_with_hash

            #Tweets with Videos
            tweets_with_videos = tweetsdata[tweetsdata["video"]==1].shape[0]
            tweets_with_videos_perc = round(tweets_with_videos/total_tweets,2)*100

            #Tweets with Image
            tweetsdata["isimage"] = tweetsdata["photos"].apply(lambda x: 1 if len(x)>0 else 0)
            tweets_with_images = tweetsdata[tweetsdata["isimage"]==1].shape[0]
            tweets_with_images_perc = round((tweets_with_images/total_tweets),2)*100


            #Figure 1 Tweets vs date

        
            fig1 = go.Figure(
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


            #Figure 2 Language analysis Pie chart

            fig2 = go.Figure(data=[go.Pie(labels=list(tweets_lang_count_dic.keys()), values=list(tweets_lang_count_dic.values()),hole=0.3,textinfo='label+percent',
                                        )],
                        layout = dict(
                        font = dict(
            #                                     family =  'Raleway',
                                                size =  16,
            #                                     color = '#7f7f7f'
                                    )))
            fig2.update_layout(title_text='Language Distribution of Tweets', title_x=0.5,title_y=0.1)


            #Weekdays analysis

            #1 - Monday
            weekday_tweets = tweetsdata.groupby("day").count()["datetime"]
            weekday_tweets_dict = {k:v for k,v in zip(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],list(weekday_tweets))}



            fig3 = go.Figure(data=[go.Bar(
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
            fig3.update_layout(title_text='Distribution of Tweets ()', title_x=0.5)


            #Teets vs Hours analysis
            hour_tweets_dict = {k:v for k,v in zip(range(24),tweetsdata.groupby("hour").count()["datetime"])}

            fig4 = go.Figure(
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



            fig5 = go.Figure(data=[go.Bar(
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
            fig5.update_layout(title_text='Frequency of Hashtags Used', title_x=0.5)

            #Figure 6

            tweetsdata_hashtags_grp_engag = tweetsdata_hashtags.groupby("hashtags").sum().sort_values(by="engagement").tail(10)["engagement"]


            fig6 = go.Figure(data=[go.Bar(
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
            fig6.update_layout(title_text='Hashtags with Most engagement (Retweets + Replies)', title_x=0.5)


            #Figure 7

            tweetsdata_hashtags_grp_likes = tweetsdata_hashtags.groupby("hashtags").sum().sort_values(by="nlikes").tail(10)["nlikes"]


            fig7 = go.Figure(data=[go.Bar(
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
            fig7.update_layout(title_text='Hashtags with Most number of likes', title_x=0.5)
            #**************************************************************************************************


            print(queryval)
            return ({'display': 'block'},[],fig1,fig2,fig3,fig4,fig5,fig6,fig7 )
            # except Exception as e:
            #     print(e)
            #     return (dash.no_update,["No user exists"],dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update)
        else:
            return (dash.no_update,["Enter Username"],dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update)
    else:
        # n_clicks,dropdown,queryval,start_date,end_date,limit_value
        c = twint.Config()

        c.Search = queryval
        c.Limit = limit_value
        c.Pandas = True

        twint.run.Search(c)

        
        querydata = twint.storage.panda.Tweets_df


        querydata = querydata[["date","place","tweet",'language','hashtags','user_id',"username",'day','hour','nlikes','nreplies','nretweets']]
        querydata["engagement"] = querydata["nreplies"]+querydata["nretweets"]

        querydata["date"] = pd.to_datetime(querydata["date"])
        querydata.rename(columns={'date':"datetime"},inplace=True)
        querydata["date"] = querydata["datetime"].dt.date


        querydata.drop("datetime",axis=1,inplace=True)


        
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
        
        #ENGAGED TWEETS
        engaged_tweetsdata = querydata[querydata["engagement"]>avg_engagement]

        engaged_tweetsdata_exploded = engaged_tweetsdata.explode("hashtags")
        top10_hash = engaged_tweetsdata.explode("hashtags").dropna().groupby("hashtags").count().sort_values("tweet",ascending=False).head(10)["tweet"]

        

if __name__ == "__main__":
    app.run_server()
