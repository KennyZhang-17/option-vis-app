import altair as alt
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import json
from scipy import stats
import sys
# import panel as pn
# pn.extension('vega')
alt.data_transformers.disable_max_rows()

import datetime as dt
date_min = dt.date(2022,4,18)
date_max = dt.date(2022,5,3)


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


def optionpriceRow(option,stockprice,date):
    # stockprice: float
    # date: in dt.date or datetime

    secondsToExpiration = (option.expirationDate / 1000 - int(date.strftime('%s')))
    daysToExpiration = secondsToExpiration / (24*3600)
    return optionprice(option.putCall,stockprice,option.strikePrice,daysToExpiration,option.volatility)

def OptionPriceVis(options,stockprice,date):
    """
    options: dataframe of options (pandas dataframe)
    stockprice: prediction of the underlying stock price (number)
    date: prediction date (dt.date or dt.datetime)
    """

    options['ExpectedPrice'] = options.apply(lambda df: optionpriceRow(df,stockprice,date),axis=1)


    options['Num'] = options.apply(lambda df: 1,axis=1)


    expected = alt.Chart(options).mark_bar(opacity=0.5).encode(
        y='symbol:N',
        x='ExpectedPrice:Q',
        tooltip = ['description','ExpectedPrice','mark'],
    )

    putCall_color = alt.Color('putCall:N',scale=alt.Scale(domain=['PUT','CALL'],range=['red','green']))


    mark = alt.Chart(options).mark_tick(thickness=3).encode(
        y='symbol:N',
        x='mark:Q',
        color = putCall_color
    )

    return mark + expected

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
        priceRange = [(low + step*i) for i in np.arange(0,stepNum+1,1)]
        options['stockPrice'] = options.apply(lambda df:priceRange,axis=1)
        newdf = options.explode('stockPrice')
        return newdf


    optionss = option_stockprice_DF(centerPrice,options)

    optionss['ExpectedPrice'] = optionss.apply(lambda df: optionpriceRow(df,df.stockPrice,date),axis=1)

    optionss['Num'] = options.apply(lambda df: 1,axis=1) # number of options for the position

    optionss['Return'] = optionss.apply(lambda df: (df.ExpectedPrice - df.mark)*df.Num,axis=1)

    indivual = alt.Chart(optionss).mark_line(thickness=20).encode(
        x='stockPrice:Q',
        y='Return:Q',
        color='symbol:N',
        tooltip = ['description','Return','Num']
    )

    total = alt.Chart(optionss).mark_line(color='black',thickness=40).encode(
        x='stockPrice:Q',
        y='sum(Return):Q',
    )

    res = indivual + total

    res = alt.concat(res).properties(
        title=alt.TitleParams(
            ['black line is the overall PnL'],
            baseline='bottom',
            orient='bottom',
            anchor='end',
            fontWeight='normal',
            fontSize=10,

            )
    )

    return res
def Greeks_Vis(options,stockprice,date):
    """
    options: dataframe of options (pandas dataframe)
    stockprice: prediction of the underlying stock price (number)
    date: prediction date (dt.date or dt.datetime)
    """


    def daysLeft(expiration,date):
        #  exipration: in Epoch ms
        # date: in dt.date or datetime
        secondsToExpiration = (expiration / 1000 - int(date.strftime('%s')))
        daysToExpiration = secondsToExpiration / (24*3600)
        return daysToExpiration

    def greeks_daily(name,type,S,K,tau,sig,dft=0):
        if name=='delta':
            res = optionprice(type,S,K,tau,sig,dft)-optionprice(type,S+1,K,tau,sig,dft)
        elif name=='theta':
            res = optionprice(type,S,K,tau,sig,dft)-optionprice(type,S,K,tau+1,sig,dft)
        else:
            sys.exit("option type error")
        return res


    options['Num'] = options.apply(lambda df: 1,axis=1)

    options['delta_daily'] = options.apply(lambda df: greeks_daily('delta',df.putCall,stockprice,df.strikePrice,daysLeft(df.expirationDate,date),df.volatility),axis=1)
    options['theta_daily'] = options.apply(lambda df: greeks_daily('theta',df.putCall,stockprice,df.strikePrice,daysLeft(df.expirationDate,date),df.volatility),axis=1)


    delta_vis = alt.Chart(options).mark_bar(opacity=0.5).encode(
        y='symbol:N',
        x='delta:Q',
        tooltip = ['description','delta'],
    )

    theta_vis = alt.Chart(options).mark_bar(opacity=0.5).encode(
        y='symbol:N',
        x='theta:Q',
        tooltip = ['description','theta'],
    )

    return delta_vis & theta_vis

def plot_altair(date=dt.date(2022,4,21),name="SPY"):
    path=os.path.join(os.path.dirname(__file__), '{}/{}{}{}'.format(name, name, date,'close'))
    #df = pd.read_csv('os.path.dirname(__file__)/{}/{}{}{}'.format(name, name, date,'close'))
    df=pd.read_csv(path)
    chart = alt.Chart(df)

    class OptionChart:
        def __init__(self,chart):
            self.chart = chart

        def add_expiration_selector(self):
            pass

    # daysToExpiration selector and legend




    def add_expiration_selector(chart):
        expiration_selector = alt.selection_multi(fields=['daysToExpiration'],on='mouseover', nearest=True)

        legend = alt.Chart(df).mark_square(size=300).encode(
            alt.X('daysToExpiration:O',axis=alt.Axis(labelAngle=0)),
            color = alt.condition(expiration_selector,alt.Color('daysToExpiration:O',legend=None),alt.value('lightgray'))
        ).add_selection(expiration_selector)
        return legend | chart.transform_filter()

    # Simplify coding

    putCall_color = alt.Color('putCall:N',legend=None,scale=alt.Scale(domain=['PUT','CALL'],range=['red','green']))

    chart = alt.Chart(df)#.encode(color=putCall_color)

    strike_as_x = chart.encode(alt.X('strikePrice:Q'))

    theoPrice = strike_as_x.mark_line().encode(alt.Y('theoreticalOptionValue:Q'))
    markPrice = strike_as_x.mark_point().encode(alt.Y('mark:Q'))
    priceRange = strike_as_x.mark_area(opacity=0.3).encode(alt.Y('lowPrice:Q'),alt.Y2('highPrice:Q'))

    prices = theoPrice

    # Expiration selector

    expiration_selector = alt.selection_multi(fields=['daysToExpiration'],on='mouseover', nearest=True)

    expiration_legend = chart.mark_square(size=300).encode(
        alt.X('daysToExpiration:O',axis=alt.Axis(labelAngle=0)),
        color = alt.condition(expiration_selector,alt.Color('daysToExpiration:O',legend=None),alt.value('lightgray'))
    ).add_selection(expiration_selector)

    # Strike selector

    brush_x = alt.selection_interval(
        encodings=['x'] # limit selection to x-axis (year) values
    )

    strikes = chart.mark_tick().add_selection(
        brush_x
    ).encode(
        alt.X('strikePrice:Q'),
        color = alt.condition(brush_x,alt.Color('strikePrice:Q',legend=None),alt.value('lightgray'))
    )

    # PutCall selector

    putOrCall_selector = alt.selection_multi(fields=['putCall'])
    putOrCall_selector_color = alt.condition(putOrCall_selector,
                    putCall_color,
                    alt.value('lightgray'))

    legend_putOrCall = chart.mark_circle(size=300).encode(
        alt.Y('putCall:N',axis=alt.Axis(orient='right')),
        color = putOrCall_selector_color
    ).add_selection(
        putOrCall_selector
    )

    def Selection_putCall(chart):
        return legend_putOrCall & chart.transform_filter(putOrCall_selector)

    # legend_putOrCall & strikes & expiration_legend & prices.encode(putCall_color).transform_filter(brush_x & expiration_selector & putOrCall_selector)



    brush = alt.selection_interval()
    strikesNexpiration = alt.Chart(df).mark_tick().add_selection(brush).encode(
        alt.X('strikePrice:Q'),
        alt.Y('daysToExpiration:O',sort='descending'),
        color = alt.condition(brush,alt.Color('strikePrice:Q',legend=None),alt.value('lightgray'))
    )

    def Selection_strikeAndExpiration(chart):
        return strikesNexpiration | chart.transform_filter(brush)



    tooltip = ['description','daysToExpiration','mark','totalVolume','delta','theta','gamma','openInterest']


    optionSelect = alt.selection_multi()
    optionSelect_color = alt.condition(optionSelect,
                    putCall_color,
                    alt.value('lightgray'))

    markPrice = markPrice.encode(alt.Size('totalVolume'),
        alt.Opacity('daysToExpiration',sort='descending',legend=None),
        alt.X('strikePrice:Q',scale=alt.Scale(domain=brush)),
        # putCall_color,
        color = optionSelect_color,
        tooltip=tooltip
        ).add_selection(optionSelect)
    return Selection_strikeAndExpiration(Selection_putCall(markPrice)).to_html()


# Fig for selection price

def plot_selection_price(name="TSLA"):
    df = pd.read_csv('./data/{}/{}'.format(name,name+'_PriceHistory'))
    #df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')
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
# name = 'TSLA'
# # date = dt.date.today()
# date = dt.date(2022,5,1)
# df=pd.read_csv('./{}/{}{}{}'.format(name, name, date,'close'))
# # Option Selector
# chart = alt.Chart(df)#.encode(color=putCall_color)
# putCall_color = alt.Color('putCall:N',legend=None,scale=alt.Scale(domain=['PUT','CALL'],range=['red','green']))
# width = 500
#
# buy_selector = alt.selection_multi(name="buy")
#
# buycolor = alt.condition(buy_selector,
#                 putCall_color,
#                 alt.value('lightgray'))
# buy = chart.mark_point(size=150).encode(
#     alt.X('strikePrice:Q'),
#     alt.Y('putCall:N'),
#     color = buycolor
#     ).add_selection(buy_selector).interactive().properties(width=width)
# vega_pane = pn.pane.Vega(buy, debounce=10)


def expiration_price(name="TSLA",callput="PUT"):
    date = dt.date(2022,5,1)
    df = pd.read_csv('./data/{}/{}{}{}'.format(name, name, date,'close'))
    df = df[df["putCall"]==callput]
    df['expirationDate']=[datetime.fromtimestamp(i/1000) for i in df['expirationDate']]
    df = df[df["expirationDate"].dt.date<dt.date(2023,1,1)]
    fig = px.scatter(df, x="strikePrice", y="expirationDate")
    fig.update_layout(clickmode='event+select')
    return fig

# Initialize
#fig=plot_selection_price(name="TSLA")

# Fig for results
name = 'TSLA'
# date = dt.date.today()
date = dt.date(2022,5,26)
df = pd.read_csv('./data/{}/{}{}{}'.format(name, name, date,'close'))

chart = alt.Chart(df)
n = 1

options = df.sample(n)

stockprice = 700

date = dt.date(2022,6,4)
def pricevis(options,stockprice,date):
    return OptionPriceVis(options,stockprice,date).to_html()

def pnlvis(options,stockprice,date):
    return Option_PnL_Vis(options=options,date=date,centerPrice=stockprice).to_html()

def greekvis(options,stockprice,date):
    return Greeks_Vis(options,stockprice,date).to_html()

def filter_option(clickData,name,putcall):
    date = dt.date(2022,5,1)
    df = pd.read_csv('./data/{}/{}{}{}'.format(name, name, date,'close'))
    #df['expirationDate2']=[datetime.fromtimestamp(i/1000).date() for i in df['expirationDate']]
    expir=[datetime.fromtimestamp(i/1000).date() for i in df['expirationDate']]
    x=clickData['x']
    y=clickData['y']
    options=df[(df['strikePrice']==x) & ([(str(i) in y) for i in expir] )]
    options=options[options['putCall']==putcall]
    return options
# def get_price(clickDate):
#     datetime.strptime(click['y'], "%Y-%m-%d %H:%M").date()
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server


app.layout = html.Div([
        html.H1('Option Visualization Tool'),
        html.P('Developed by Yang Yu and Kenny Zhang from University of Washington'),
        dcc.Dropdown(
            id='putcall', value='PUT',
            options=[
            {'label': 'PUT', 'value': 'PUT'},
            {'label': 'CALL', 'value': 'CALL'}]),
        dcc.Dropdown(
            id='stock', value='TSLA',
            options=[
            {'label': 'SPY', 'value': 'SPY'},
            {'label': 'TSLA', 'value': 'TSLA'}]),
        # dcc.DatePickerSingle(
        #             id="date-single",
        #             min_date_allowed=date_min,
        #             max_date_allowed=date_max,
        #             initial_visible_month=dt.date(2022, 4, 1),
        #             date=dt.date(2022, 4, 21)),
        # html.Iframe(
        #     id='scatter',
        #     style={'border-width': '0', 'width': '100%', 'height': '400px'},
        #     srcDoc=plot_altair(date=dt.date(2022, 4, 21),name='SPY')),
        html.P('Click an option for trading'),
        dcc.Graph(figure=expiration_price(name="TSLA",callput="CALL"),id="expiration_price"),
        # html.Iframe(
        #     id='option_selector',
        #     style={'border-width': '0', 'width': '100%', 'height': '400px'},
        #     srcDoc=buy.to_html()),
        html.P('Click a future price for prediction'),
        dcc.Graph(figure=plot_selection_price(name="TSLA"),id="plot"),
        html.P('Options and future price selected:'),
        dcc.Textarea(id='widget'),
        #dcc.Textarea(id='widget2'),
        dcc.Textarea(id='widget3'),
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
        ])

# @app.callback(
#     Output('scatter', 'srcDoc'),
#     Input('date-single', 'date'),
#     Input('stock', 'value')
#     #nput('pc', 'value'),
#     )
#
# def update_output(date,value):
#     return plot_altair(date,value)

@app.callback(
    Output("widget", "value"),
    Input("plot", "clickData")
)
def update_widget(clickData):
    if(clickData):
        return json.dumps({k: clickData["points"][0][k] for k in ["x", "y"]})

@app.callback(
    Output("expiration_price", "figure"),
    Input('putcall', 'value'),
    Input('stock', 'value')
)

def update_plot(a,b):
    return expiration_price(name=b,callput=a)

# @app.callback(
#     Output("widget2", "value"),
#     Input('expiration_price', 'clickData'),
#     #Input('expiration_price', 'selectedData')
# )
#
# def update_widget2(clickData):
#     #holder = []
#     if(clickData):
#         #holder.append(str(int(clickData["points"][0]["x"])))
#         return json.dumps({k: clickData["points"][0][k] for k in ["x", "y"]})
#     # if(value):
#     #     for x in value["points"]:
#     #         holder.append(json.dumps({k: x["points"][0][k] for k in ["x", "y"]}))
#     #     return str(list(set(holder)))

@app.callback(
    Output("pricevis", "srcDoc"),
    Input('stock', 'value'),
    Input('putcall','value'),
    Input('expiration_price', 'clickData'),
    Input('expiration_price', 'selectedData'),
    Input('plot','clickData')
    #Input('expiration_price', 'selectedData')
)
def update_pricevis(name, putcall,click_option,select_option,click_price):
    # if(click_option):
    #     click_option=json.loads(json.dumps({k: click_option["points"][0][k] for k in ["x", "y"]}))
    #     option=filter_option(click_option,name,putcall)
    if(select_option):
        holder=[]
        for x in select_option["points"]:
            holder.append({k:x[k] for k in ["x", "y"]})
        option=filter_option(holder[0],name,putcall)
        for i in holder[1:len(holder)]:
            s=filter_option(i,name,putcall)
            option=pd.concat([option,s])
    if((select_option is not None) & (click_price is not None)):
        click_price=json.loads(json.dumps({k: click_price["points"][0][k] for k in ["x", "y"]}))
        price=click_price['y']
        date_select=datetime.strptime(click_price['x'], "%Y-%m-%d %H:%M").date()
        return pricevis(option,price,date_select)
    return pricevis(options,stockprice,date)
# def update_pricevis(name, putcall,click_option,click_price):
#     if((click_option is not None) & (click_price is not None)):
#         click_option=json.loads(json.dumps({k: click_option["points"][0][k] for k in ["x", "y"]}))
#         click_price=json.loads(json.dumps({k: click_price["points"][0][k] for k in ["x", "y"]}))
#         option=filter_option(click_option,name,putcall)
#         price=click_price['y']
#         date_select=datetime.strptime(click_price['x'], "%Y-%m-%d %H:%M").date()
#         return pricevis(option,price,date_select)
#     return pricevis(options,stockprice,date)

@app.callback(
    Output("pnlvis", "srcDoc"),
    Input('stock', 'value'),
    Input('putcall','value'),
    Input('expiration_price', 'clickData'),
    Input('expiration_price', 'selectedData'),
    Input('plot','clickData')
    #Input('expiration_price', 'selectedData')
)
def update_pnlvis(name, putcall,click_option,select_option,click_price):
    # if(click_option):
    #     click_option=json.loads(json.dumps({k: click_option["points"][0][k] for k in ["x", "y"]}))
    #     option=filter_option(click_option,name,putcall)
    if(select_option):
        holder=[]
        for x in select_option["points"]:
            holder.append({k:x[k] for k in ["x", "y"]})
        option=filter_option(holder[0],name,putcall)
        for i in holder[1:len(holder)]:
            s=filter_option(i,name,putcall)
            option=pd.concat([option,s])
    if((select_option is not None) & (click_price is not None)):
        click_price=json.loads(json.dumps({k: click_price["points"][0][k] for k in ["x", "y"]}))
        price=click_price['y']
        date_select=datetime.strptime(click_price['x'], "%Y-%m-%d %H:%M").date()
        return pnlvis(option,price,date_select)
    return pnlvis(options,stockprice,date)
# def update_pnlvis(name, putcall,click_option,click_price):
#     if((click_option is not None) & (click_price is not None)):
#         click_option=json.loads(json.dumps({k: click_option["points"][0][k] for k in ["x", "y"]}))
#         click_price=json.loads(json.dumps({k: click_price["points"][0][k] for k in ["x", "y"]}))
#         option=filter_option(click_option,name,putcall)
#         price=click_price['y']
#         date_select=datetime.strptime(click_price['x'], "%Y-%m-%d %H:%M").date()
#         return pnlvis(option,price,date_select)
#     return pnlvis(options,stockprice,date)

@app.callback(
    Output("greekvis", "srcDoc"),
    Input('stock', 'value'),
    Input('putcall','value'),
    Input('expiration_price', 'clickData'),
    Input('expiration_price', 'selectedData'),
    Input('plot','clickData')
    #Input('expiration_price', 'selectedData')
)

def update_greekvis(name, putcall,click_option,select_option,click_price):
    # if(click_option):
    #     click_option=json.loads(json.dumps({k: click_option["points"][0][k] for k in ["x", "y"]}))
    #     option=filter_option(click_option,name,putcall)
    if(select_option):
        holder=[]
        for x in select_option["points"]:
            holder.append({k:x[k] for k in ["x", "y"]})
        option=filter_option(holder[0],name,putcall)
        for i in holder[1:len(holder)]:
            s=filter_option(i,name,putcall)
            option=pd.concat([option,s])
    if((select_option is not None) & (click_price is not None)):
        click_price=json.loads(json.dumps({k: click_price["points"][0][k] for k in ["x", "y"]}))
        price=click_price['y']
        date_select=datetime.strptime(click_price['x'], "%Y-%m-%d %H:%M").date()
        return greekvis(option,price,date_select)
    return greekvis(options,stockprice,date)

@app.callback(
    Output("widget3", "value"),
    #Input('expiration_price', 'clickData'),
    Input('expiration_price', 'selectedData')
)

def update_widget3(selectedData):
    #holder = []
    if(selectedData):
        #holder.append(str(int(clickData["points"][0]["x"])))
        #return selectedData
        holder=[]
        for x in selectedData["points"]:
            holder.append({k:x[k] for k in ["x", "y"]})
        #return json.dumps({k: selectedData["points"][0][k] for k in ["x", "y"]})
        return str(holder)
    # if(value):
    #     for x in value["points"]:
    #         holder.append(json.dumps({k: x["points"][0][k] for k in ["x", "y"]}))
    #     return str(list(set(holder)))

@app.callback(
    Output("plot", "figure"),
    Input('stock', 'value')
)

def update_plot(name):
    return plot_selection_price(name)


if __name__ == '__main__':
    app.run_server(debug=True)
