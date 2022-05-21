import altair as alt
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import os
import pandas as pd
import plotly.express as px
alt.data_transformers.disable_max_rows()

import datetime as dt
date_min = dt.date(2022,4,18)
date_max = dt.date(2022,5,3)


date=dt.date(2022,4,21)
name="SPY"
path=os.path.join(os.path.dirname(__file__), '{}/{}{}{}'.format(name, name, date,'close'))
    #df = pd.read_csv('os.path.dirname(__file__)/{}/{}{}{}'.format(name, name, date,'close'))
df=pd.read_csv(path)

fig = px.scatter(df, x="gamma", y="delta")

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

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server


app.layout = html.Div([
        # dcc.Dropdown(
        #     id='bs', value='Buy',
        #     options=[
        #     {'label': 'Buy', 'value': 'Buy'},
        #     {'label': 'Sell', 'value': 'Sell'}]),
        dcc.Dropdown(
            id='stock', value='SPY',
            options=[
            {'label': 'SPY', 'value': 'SPY'},
            {'label': 'TSLA', 'value': 'TSLA'}]),
        dcc.DatePickerSingle(
                    id="date-single",
                    min_date_allowed=date_min,
                    max_date_allowed=date_max,
                    initial_visible_month=dt.date(2022, 4, 1),
                    date=dt.date(2022, 4, 21)),
        html.Iframe(
            id='scatter',
            style={'border-width': '0', 'width': '100%', 'height': '400px'},
            srcDoc=plot_altair(date=dt.date(2022, 4, 21),name='SPY')),
        dcc.Graph(figure=fig,id="plot"),
        dcc.Textarea(id='widget')])

@app.callback(
    Output('scatter', 'srcDoc'),
    Input('date-single', 'date'),
    Input('stock', 'value')
    #nput('pc', 'value'),
    )

def update_output(date,value):
    return plot_altair(date,value)
    
@app.callback(
    Output("widget", "value"),
     [Input("plot", "selectedData"), Input("plot", "clickData")]
)
def update_widget(value,clickData):
    holder = []
    if(clickData):
        holder.append(str(clickData["points"][0]['x'])+","+str(clickData["points"][0]['y']))
    if(value):
        for x in value["points"]:
            holder.append(str(x['x'])+str(x['y']))
    return str(holder)




if __name__ == '__main__':
    app.run_server(debug=True)
