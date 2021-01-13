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