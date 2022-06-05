import altair as alt
import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output, State
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import json
from scipy import stats
import sys

#import dash_bootstrap_components as dbc
# import panel as pn
# pn.extension('vega')
alt.data_transformers.disable_max_rows()

import datetime as dt
#date_min = dt.date(2022,4,18)
#date_max = dt.date(2022,5,3)


# date=dt.date(2022,4,21)
# name="SPY"
# path=os.path.join(os.path.dirname(__file__), '{}/{}{}{}'.format(name, name, date,'close'))
#     #df = pd.read_csv('os.path.dirname(__file__)/{}/{}{}{}'.format(name, name, date,'close'))
# df=pd.read_csv(path)

#fig = px.scatter(df, x="gamma", y="delta")
def optionprice(type,S,K,tau,sig,dft=0):
    '''
    type : option type: 'CALL' or 'PUT'
    S: stock price: float/int
    K: strike price: float/int
    tau: days left to expiration: float/int
    sig: volatility: float/int, without %
    dft: drift rate in years (due to bias or inflation): float, default=0, best match for TSLA around 0.02
    '''

    tau = max(tau,0)/ 365 # turn in to years, avoid negative
    sig = sig/100 # add %

    if type=='CALL':
        if S == 0:  # this is to avoid log(0) issues
            return 0.0
        elif tau == 0 or sig == 0:  # this is to avoid 0/0 issues
            return max(S - K, 0)
        else:
            d = (np.log(S / K) + dft * tau) / (sig * np.sqrt(tau))

            d1 = d + sig * np.sqrt(tau) / 2

            d2 = d - sig * np.sqrt(tau) / 2

            price =  np.exp(dft * tau) * S * stats.norm.cdf(d1, 0.0, 1.0) - K * stats.norm.cdf(d2, 0.0, 1.0)
        return price

    elif type == 'PUT':
        if S == 0:  # this is to avoid log(0) issues
            return 0.0
        elif tau == 0 or sig == 0:  # this is to avoid 0/0 issues
            return max(K - S, 0)
        else:
            d = (np.log(S / K) + dft * tau) / (sig * np.sqrt(tau))

            d1 = -d + sig * np.sqrt(tau) / 2

            d2 = -d - sig * np.sqrt(tau) / 2

            price = K * stats.norm.cdf(d1, 0.0, 1.0) - np.exp(dft * tau) * S * stats.norm.cdf(d2, 0.0, 1.0)

            return price
    else:
        sys.exit("option type error")

def greeks_theo(name,type,S,K,tau,sig,dft=0):
    if name=='delta':
        res = optionprice(type,S+1,K,tau,sig,dft)-optionprice(type,S,K,tau,sig,dft) #if tau>0 else 0
    elif name=='theta':
        res = optionprice(type,S,K,tau-1,sig,dft)-optionprice(type,S,K,tau,sig,dft) #if tau>0 else 0
    elif name=='gamma':
        res = optionprice(type,S+1,K,tau,sig,dft)+optionprice(type,S-1,K,tau,sig,dft)-2*optionprice(type,S,K,tau,sig,dft) #if tau>0 else 0
    else:
        sys.exit("option type error")
    return res * 100


def daysLeft(expiration,date):
    #  exipration: in Epoch ms
    # date: in dt.date or datetime
    secondsToExpiration = (expiration / 1000 - int(date.strftime('%s')))
    daysToExpiration = secondsToExpiration / (24*3600)
    return daysToExpiration

def optionpriceRow(option,stockprice,date):
    # stockprice: float
    # date: in dt.date or datetime

    secondsToExpiration = (option.expirationDate / 1000 - int(date.strftime('%s')))
    daysToExpiration = secondsToExpiration / (24*3600)
    return optionprice(option.putCall,stockprice,option.strikePrice,daysToExpiration,option.volatility)


def linear_price_range(latestPrice,percent=0.01,stepNum=20):
    low = latestPrice*((1-percent))
    high = latestPrice*((1+percent))
    step = (high-low) / stepNum
    priceRange = [ low + step*num for num in range(stepNum+1)]
    return priceRange

def option_stockprice_dates_DF(priceRange,dateRange,options):
    # https://stackoverflow.com/questions/42168103/how-to-expand-flatten-pandas-dataframe-efficiently
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.explode.html

    """
    priceRange: list
    dateRange: list in Epoch(s)
    options: dataframe
    """
    options['stockPrice'] = options.apply(lambda df:priceRange,axis=1)
    options['dates'] = options.apply(lambda df:dateRange,axis=1)
    newdf = options.explode('stockPrice').explode('dates')
    return newdf
def Greeks_Vis(options,stockprice,date):
    """
    options: dataframe of options (pandas dataframe)
    stockprice: prediction of the underlying stock price (number)
    date: prediction date (dt.date or dt.datetime)
    """

    options['delta_theo'] = options.apply(lambda df: df.Num * greeks_theo('delta',df.putCall,stockprice,df.strikePrice,daysLeft(df.expirationDate,date),df.volatility),axis=1)
    options['theta_theo'] = options.apply(lambda df: df.Num * greeks_theo('theta',df.putCall,stockprice,df.strikePrice,daysLeft(df.expirationDate,date),df.volatility),axis=1)
    options['gamma_theo'] = options.apply(lambda df: df.Num * greeks_theo('gamma',df.putCall,stockprice,df.strikePrice,daysLeft(df.expirationDate,date),df.volatility),axis=1)

    width = 280

    delta_vis = alt.Chart(options).mark_bar().encode(
        alt.Opacity('Num:O'),
        y='symbol:N',
        x='delta_theo:Q',
        # tooltip = ['description','delta_theo','Num'],
    ).properties(width=width)
    nearest1 = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['symbol'])
    tooltip_selectors1 = alt.Chart(options).mark_point().encode(
        y="symbol:N",
        opacity=alt.value(0),
        tooltip = ['description:N','delta_theo:Q','Num:Q'],
    ).add_selection(
        nearest1
    )
    delta_vis = (delta_vis + tooltip_selectors1)


    theta_vis = alt.Chart(options).mark_bar().encode(
        alt.Opacity('Num:O'),
        y='symbol:N',
        x='theta_theo:Q',
        tooltip = ['description','theta_theo','Num'],
    ).properties(width=width)
    nearest2 = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['symbol'])
    tooltip_selectors2 = alt.Chart(options).mark_point().encode(
        y="symbol:N",
        opacity=alt.value(0),
        tooltip = ['description:N','theta_theo:Q','Num:Q'],
    ).add_selection(
        nearest2
    )
    theta_vis = (theta_vis+tooltip_selectors2)

    gamma_vis = alt.Chart(options).mark_bar(opacity=0.5).encode(
        alt.Opacity('Num:O'),
        y='symbol:N',
        x='gamma_theo:Q',
        tooltip = ['description','gamma_theo','Num'],
    ).properties(width=width)
    nearest3 = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['symbol'])
    tooltip_selectors3 = alt.Chart(options).mark_point().encode(
        y="symbol:N",
        opacity=alt.value(0),
        tooltip = ['description:N','gamma_theo:Q','Num:Q'],
    ).add_selection(
        nearest3
    )
    gamma_vis = (gamma_vis+ tooltip_selectors3)

    return delta_vis | theta_vis | gamma_vis


def Option_PnL_Vis(options,date,centerPrice):
    """
    options: dataframe of options (pandas dataframe)
    date: date of PnL  (dt.date or dt.datetime)
    centerPrice: The visualization will draw the PnL diagram around centerPrice
    """



    def option_stockprice_DF(centerPrice,options,stepNum=500,percent=0.8):
        # https://stackoverflow.com/questions/42168103/how-to-expand-flatten-pandas-dataframe-efficiently
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.explode.html

        """
        Create a dataframe of options with varying stockPrice
        """
        low = centerPrice*((1-percent))
        high = centerPrice*((1+percent))
        step = (high-low) / stepNum
        priceRange = [ low + step*num for num in range(stepNum+1)]
        options['stockPrice'] = options.apply(lambda df:priceRange,axis=1)
        newdf = options.explode('stockPrice')
        return newdf


    optionss = option_stockprice_DF(centerPrice,options)

    optionss['ExpectedPrice'] = optionss.apply(lambda df: optionpriceRow(df,df.stockPrice,date),axis=1)

    # optionss['Num'] = options.apply(lambda df: 1,axis=1) # number of options for the position

    optionss['Return'] = optionss.apply(lambda df: (df.ExpectedPrice - df.mark)*df.Num,axis=1)

    indivual = alt.Chart(optionss).encode(
        x='stockPrice:Q',
        y='Return:Q',
        color='symbol:N',
        tooltip = ['description','stockPrice','Return','Num']
    )

    # https://altair-viz.github.io/gallery/multiline_highlight.html

    highlight = alt.selection(type='single', on='mouseover',
                        fields=['symbol'], nearest=True)

    points = indivual.mark_circle().encode(
        opacity=alt.value(0)
    ).add_selection(
        highlight
    )

    lines = indivual.mark_line().encode(size=alt.condition(~highlight, alt.value(1), alt.value(3)))

    res = (points+lines).interactive()

    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html
    # https://stackoverflow.com/questions/20490274/how-to-reset-index-in-a-pandas-dataframe
    totalReturnDF = optionss[['stockPrice','Return']].groupby(['stockPrice']).sum().reset_index()


    total = alt.Chart(totalReturnDF).mark_line(color='black').encode(
        x='stockPrice',
        y=alt.Y('Return:Q',title='Total Return'),
    )

    # https://altair-viz.github.io/gallery/multiline_tooltip.html
    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['stockPrice'], empty='none')

    selectors = alt.Chart(totalReturnDF).mark_point().encode(
        x='stockPrice:Q',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )

    points = total.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    text = total.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, 'Return:Q', alt.value(' '))
    )

    rules = alt.Chart(totalReturnDF).mark_rule(color='gray').encode(
        x='stockPrice:Q'
    ).transform_filter(
        nearest
    )

    total = alt.layer(total, selectors, rules, points, text)


    res = res.properties(width=600) | total.properties(width=600).interactive()

    return res

def OptionPriceVis(options,stockprice,date):
    """
    options: dataframe of options (pandas dataframe)
    stockprice: prediction of the underlying stock price (number)
    date: prediction date (dt.date or dt.datetime)
    """

    options['ExpectedPrice'] = options.apply(lambda df: optionpriceRow(df,stockprice,date),axis=1)

    options['ExpectedReturn'] = options.apply(lambda df: df.Num * (df.ExpectedPrice - df.mark),axis=1)

    expected = alt.Chart(options).mark_bar(opacity=0.5).encode(
        alt.Opacity('Num:O'),
        y='symbol:N',
        x='ExpectedPrice:Q',
    )

    putCall_color = alt.Color('putCall:N',scale=alt.Scale(domain=['PUT','CALL'],range=['red','green']))

    mark = alt.Chart(options).mark_tick(thickness=3).encode(
        y='symbol:N',
        x='mark:Q',
        color = putCall_color
    )



    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['symbol'])
    tooltip_selectors1 = alt.Chart().mark_point().encode(
        y="symbol:N",
        opacity=alt.value(0),
        tooltip = ['description:N','ExpectedPrice:Q','mark:Q','Num:Q'],
    ).add_selection(
        nearest
    )

    # longOrShort = alt.condition(alt.datum.Num>0,alt.value(''), alt.value('red'))
    longOrShort = 'ifTest ? thenValue : '

    returnChart = alt.Chart(options).mark_bar().encode(
        alt.Opacity('Num:O'),
        y='symbol:N',
        x='ExpectedReturn:Q',

        # color=alt.condition(alt.datum.ExpectedReturn>0,alt.value('green'), alt.value('red'))
    )

    nearest2 = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['symbol'])

    tooltip_selectors2 = alt.Chart(options).mark_point().encode(
        y="symbol:N",
        opacity=alt.value(0),
        tooltip = ['description:N','ExpectedReturn:Q','Num:Q'],
    ).add_selection(
        nearest2
    )





    return (mark+expected+tooltip_selectors1) | (returnChart + tooltip_selectors2)

def Greek_Table_Vis(options,stockprice,days=30,greek='delta'):
    """
    Visualize future options greeks in {days} days
    options: dateframe
    days: positive integer
    priceRange: list of prices
    greek: greek name to visualize, 'delta','theta','gamma'
    """
    priceRange = linear_price_range(stockprice,stepNum=20,percent=0.2)
    dateRange = [1000*int((dt.date.today()+dt.timedelta(days=i)).strftime('%s')) for i in range(days)]


    df_greeksTable = option_stockprice_dates_DF(priceRange,dateRange,options)


    def calculateGreeks(options):
        for greek in ['delta','theta','gamma']:
            options['{}_theo'.format(greek)] = options.apply(lambda df: df.Num*greeks_theo(greek,df.putCall,df.stockPrice,df.strikePrice,daysLeft(df.expirationDate,dt.datetime.fromtimestamp(df.dates/1000)),df.volatility),axis=1)

    calculateGreeks(df_greeksTable)

    GreeksDF = df_greeksTable[['stockPrice','delta_theo','theta_theo','gamma_theo','dates']].groupby(['stockPrice','dates']).sum().reset_index()


    # https://altair-viz.github.io/user_guide/times_and_dates.html
    # https://altair-viz.github.io/user_guide/transform/timeunit.html#user-guide-timeunit-transform
    a = alt.Chart(GreeksDF).mark_rect().encode(
        alt.X('monthdate(dates):O'),
        alt.Y('stockPrice:O',scale=alt.Scale(zero=False),sort='descending'),
        alt.Color('{}_theo:Q'.format(greek), scale=alt.Scale(scheme='purpleblue')),
        tooltip=['dates:T','stockPrice','delta_theo','theta_theo','gamma_theo','dates']
    ).properties(width=600)

    return a

# Plotly fig for selection price
def plot_selection_price(name="TSLA"):
    df = pd.read_csv('./data/{}/{}'.format(name,name+'_PriceHistory'))
    df['datetime']=[datetime.fromtimestamp(i/1000) for i in df['datetime']]
    fig = go.Figure(data=[go.Candlestick(x=df['datetime'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'])])
    fig=fig.update_layout(xaxis_rangeslider_visible=False)
    prange=list(df['datetime'].values)+[max(df['datetime'])+dt.timedelta(days=i+1) for i in range(30)]
    fig = fig.add_traces(px.scatter(
            x=np.repeat(prange,500), y=np.tile(np.linspace(np.min(df["low"])-np.std(df["low"]),
            np.max(df['high'])+np.std(df["high"]), len(prange)), 500)
        )
        .update_traces(marker_color="rgba(0,0,0,0)")
        .data
    )
    return fig

# Plotly scatter plot to select options
def expiration_price(name="TSLA",callput="PUT"):
    df = pd.read_csv('./data/{}/{}{}{}'.format(name, name, date,'close'))
    df = df[df["putCall"]==callput]
    df['expirationDate']=[datetime.fromtimestamp(i/1000) for i in df['expirationDate']]
    fig = px.scatter(df, x="strikePrice", y="expirationDate",title = "Select "+callput)
    fig.update_yaxes(type='category',tickformat="%Y\n%b")#tickformat not functioning in category type
    fig.update_layout(clickmode='event+select')
    return fig

# Initialize data for the website

# Fig for results
name = 'TSLA'
date = dt.date(2022,6,4)
df = pd.read_csv('./data/{}/{}{}{}'.format(name, name, date,'close'))

chart = alt.Chart(df)
n = 2

options = df.sample(n)
num=np.ones(len(options))
options['Num']=num
stockprice = 700


# Wrap plotting functions to return html
def pricevis(options,stockprice,date):
    return OptionPriceVis(options,stockprice,date).to_html()

def pnlvis(options,stockprice,date):
    return Option_PnL_Vis(options=options,date=date,centerPrice=stockprice).to_html()

def greekvis(options,stockprice,date):
    return Greeks_Vis(options,stockprice,date).to_html()

def greektablevis(options,stockprice):
    return Greek_Table_Vis(options,stockprice,days=30).to_html()

# Table for print selected option information
def print_options(options):
    option_table=options.iloc[:,[1,3,8,23,24,25,35]] # only display these columns
    table=dash_table.DataTable(id='option-table', data=option_table.to_dict('records'), columns=[{"name": i, "id": i} for i in option_table.columns])
    return table

# Table for input number of quantities
def quantity_table(n):
    table=dash_table.DataTable(
        id='computed-table',
        columns=[
            {'name': 'Quantity', 'id': 'input-data'},
        ],
        data=[{'input-data': i*0+1} for i in range(n)],
        editable=True,
    )
    return table

# helper function to get option from click data
def filter_option(clickData,name,putcall):
    df = pd.read_csv('./data/{}/{}{}{}'.format(name, name, date,'close'))
    expir=[datetime.fromtimestamp(i/1000).date() for i in df['expirationDate']]
    x=clickData['x']
    y=clickData['y']
    options=df[(df['strikePrice']==x) & ([(str(i) in y) for i in expir] )]
    options=options[options['putCall']==putcall]
    options['Num']=np.ones(len(options))
    return options

def get_option(name,select_option=None,select_option_put=None):
    option=None
    if(select_option):
        holder=[]
        for x in select_option["points"]:
            holder.append({k:x[k] for k in ["x", "y"]})
        option=filter_option(holder[0],name,"CALL")
        for i in holder[1:len(holder)]:
            s=filter_option(i,name,"CALL")
            option=pd.concat([option,s])
    if(select_option_put):
        holder=[]
        for x in select_option_put["points"]:
            holder.append({k:x[k] for k in ["x", "y"]})
        option_put=filter_option(holder[0],name,"PUT")
        for i in holder[1:len(holder)]:
            s=filter_option(i,name,"PUT")
            option_put=pd.concat([option_put,s])
    if((select_option is not None) & (select_option_put is not None)):
        option=pd.concat([option,option_put])
    elif(select_option_put is not None):
        option=option_put
    return option

# style sheet
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server

# App layout
app.layout = html.Div([
        html.H1('Option Visualization Tool'),
        html.P('Developed by Yang Yu and Kenny Zhang from University of Washington'),
        html.P('Select a stock for trading'),
        dcc.Dropdown(
            id='stock', value='TSLA',
            options=[
            {'label': 'SPY', 'value': 'SPY'},
            {'label': 'TSLA', 'value': 'TSLA'},
            {'label': 'AAPL', 'value': 'AAPL'},
            {'label': 'QQQ', 'value': 'QQQ'}]),
        html.P('Click option(s) for trading'),
        html.Div(className='row',children=[
        dcc.Graph(figure=expiration_price(name="TSLA",callput="CALL"),id="expiration_price",style={'display': 'inline-block'}),
        dcc.Graph(figure=expiration_price(name="TSLA",callput="PUT"),id="expiration_price_put",style={'display': 'inline-block'}),
    ]),
        html.P('Click a future price for prediction'),
        dcc.Graph(figure=plot_selection_price(name="TSLA"),id="plot"),
        html.P('You may change the quantity by typing numbers in the cell and press enter. Options selected:'),
        html.Div([
        html.Div([print_options(options)], style={'display': 'inline-block'}),
        html.Div([quantity_table(n)], style={'display': 'inline-block'}),
    ]),
        html.P('Future date and price selected:'),
        dcc.Textarea(id='widget1'),
        dcc.Textarea(id='widget2'),
        html.P('Prediction result:'),
        html.Iframe(
            id='pricevis',
            style={'border-width': '0', 'width': '100%', 'height': '200px'},
            srcDoc=pricevis(options,stockprice,date)),
        html.Iframe(
            id='pnlvis',
            style={'border-width': '0', 'width': '100%', 'height': '400px'},
            srcDoc=pnlvis(options,stockprice,date)),
        html.Iframe(
            id='greekvis',
            style={'border-width': '0', 'width': '100%', 'height': '200px'},
            srcDoc=greekvis(options,stockprice,date)),
        html.Iframe(
            id='greektablevis',
            style={'border-width': '0', 'width': '100%', 'height': '500px'},
            srcDoc=greektablevis(options,stockprice)),
        ])

# call back for display future price and date selection
@app.callback(
    Output("widget1", "value"),
    Input("plot", "clickData")
)
def update_widget1(clickData):
    if(clickData):
        return str(clickData["points"][0]["x"])

@app.callback(
    Output("widget2", "value"),
    Input("plot", "clickData")
)
def update_widget2(clickData):
    if(clickData):
        return str(clickData["points"][0]["y"])

# callback to display options (call) based on stock selected
@app.callback(
    Output("expiration_price", "figure"),
    Input('stock', 'value')
)

def update_plot(b):
    return expiration_price(name=b,callput="CALL")

# callback to display options (put) based on stock selected
@app.callback(
    Output("expiration_price_put", "figure"),
    Input('stock', 'value')
)

def update_plot(b):
    return expiration_price(name=b,callput="PUT")

# callback to update prediction plots
@app.callback(
    Output("pricevis", "srcDoc"),
    Input('stock', 'value'), # stock selected
    Input('expiration_price', 'selectedData'), # option (call) selected
    Input('expiration_price_put', 'selectedData'), # option (put) selected
    Input('plot','clickData'), # future price and expiration selected
    Input("computed-table", "data_timestamp"), # input quantities
    State('computed-table', 'data')
)
def update_pricevis(name,select_option,select_option_put,click_price,timestamp,rows):
    option=get_option(name,select_option,select_option_put)
    if((option is not None) & (click_price is not None)):
        click_price=json.loads(json.dumps({k: click_price["points"][0][k] for k in ["x", "y"]}))
        price=click_price['y']
        date_select=datetime.strptime(click_price['x'], "%Y-%m-%d %H:%M").date()
        if(rows):
            num=[]
            for row in rows:
                num.append(int(row['input-data']))
            if(len(num)==len(option)):
                option['Num']=num
        return pricevis(option,price,date_select)



@app.callback(
    Output("pnlvis", "srcDoc"),
    Input('stock', 'value'),
    Input('expiration_price', 'selectedData'),
    Input('expiration_price_put', 'selectedData'),
    Input('plot','clickData'),
    Input("computed-table", "data_timestamp"),
    State('computed-table', 'data')
)
def update_pnlvis(name,select_option,select_option_put,click_price,timestamp,rows):
    option=get_option(name,select_option,select_option_put)
    if((option is not None) & (click_price is not None)):
        click_price=json.loads(json.dumps({k: click_price["points"][0][k] for k in ["x", "y"]}))
        price=click_price['y']
        date_select=datetime.strptime(click_price['x'], "%Y-%m-%d %H:%M").date()
        if(rows):
            num=[]
            for row in rows:
                num.append(int(row['input-data']))
            if(len(num)==len(option)):
                option['Num']=num
        return pnlvis(option,price,date_select)


@app.callback(
    Output("greekvis", "srcDoc"),
    Input('stock', 'value'),
    Input('expiration_price', 'selectedData'),
    Input('expiration_price_put', 'selectedData'),
    Input('plot','clickData'),
    Input("computed-table", "data_timestamp"),
    State('computed-table', 'data')
)
def update_greekvis(name,select_option,select_option_put,click_price,timestamp,rows):
    option=get_option(name,select_option,select_option_put)
    if((option is not None) & (click_price is not None)):
        click_price=json.loads(json.dumps({k: click_price["points"][0][k] for k in ["x", "y"]}))
        price=click_price['y']
        date_select=datetime.strptime(click_price['x'], "%Y-%m-%d %H:%M").date()
        if(rows):
            num=[]
            for row in rows:
                num.append(int(row['input-data']))
            if(len(num)==len(option)):
                option['Num']=num
        return greekvis(option,price,date_select)

@app.callback(
    Output("greektablevis", "srcDoc"),
    Input('stock', 'value'),
    Input('expiration_price', 'selectedData'),
    Input('expiration_price_put', 'selectedData'),
    Input('plot','clickData'),
    Input("computed-table", "data_timestamp"),
    State('computed-table', 'data')
)
def update_greektablevis(name,select_option,select_option_put,click_price,timestamp,rows):
    option=get_option(name,select_option,select_option_put)
    if((option is not None) & (click_price is not None)):
        click_price=json.loads(json.dumps({k: click_price["points"][0][k] for k in ["x", "y"]}))
        price=click_price['y']
        if(rows):
            num=[]
            for row in rows:
                num.append(int(row['input-data']))
            if(len(num)==len(option)):
                option['Num']=num
        return greektablevis(option,price)


@app.callback(
    Output("plot", "figure"),
    Input('stock', 'value')
)

def update_plot(name):
    return plot_selection_price(name)

# callback to update option table based on option selected
@app.callback(
    Output("option-table", "data"),
    Input('stock', 'value'),
    Input('expiration_price', 'selectedData'),
    Input('expiration_price_put', 'selectedData'),
)

def update_table(name,select_option,select_option_put):
    option=get_option(name,select_option,select_option_put)
    if(option is not None):
        return option.iloc[:,[1,3,8,23,24,25,35]].to_dict(orient='records')
    return options.iloc[:,[1,3,8,23,24,25,35]].to_dict(orient='records')

# callback to update input quantity table based on option selected
@app.callback(
    Output('computed-table', 'data'),
    Input('stock', 'value'),
    Input('expiration_price', 'selectedData'),
    Input('expiration_price_put', 'selectedData'),
)
def update_quantity_table(name,select_option,select_option_put):
    n=2 # same as initialization, may not be necessary
    option=get_option(name,select_option,select_option_put)
    if(option is not None):
        n=len(option)
    return pd.DataFrame({'input-data':np.ones(n)}).to_dict(orient='records')


if __name__ == '__main__':
    app.run_server(debug=True)
