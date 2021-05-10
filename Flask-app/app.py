import os

from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt

import requests, json, re, folium
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster, HeatMapWithTime
from branca.element import Template, MacroElement

app = Flask(__name__)

app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.route('/')
def index():
    covid_timeseries = pd.read_csv(
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    travel_ban = pd.read_csv('./data/international-travel-covid.csv')
    travel_ban = travel_ban.sort_values('Day').drop_duplicates('Entity', keep='last')
    travel_ban['Entity'] = travel_ban['Entity'].replace({"United States": "US",
                                                         "Congo": "Congo (Brazzaville)",
                                                         "Democratic Republic of Congo": "Congo (Kinshasa)",
                                                         'South Korea': "Korea, South", "Taiwan": "Taiwan*",
                                                         "Timor": "Timor-Leste", "Cape Verde": "Cabo Verde"})
    covid_timeseries.loc[103, 'Country/Region'] = 'Greenland'
    travel_ban = travel_ban.rename(columns={"Entity": "country", "Day": "restrictions_update_date"})
    covid_timeseries = covid_timeseries.rename(columns={'Province/State': 'state', 'Country/Region': 'country'})
    covid_data = covid_timeseries.merge(travel_ban, on='country', how='left')
    covid_data.columns = map(str.lower, covid_data.columns)
    covid_data.drop_duplicates(inplace=True)
    covid_data.dropna(subset=["lat"], inplace=True)
    object_columns = ['state', 'code', 'restrictions_update_date']
    for column in object_columns:
        covid_data[column].fillna('', inplace=True)
    covid_data['international_travel_controls'].fillna(-1, inplace=True)
    covid_data['1/22/20'].fillna(0, inplace=True)  # replace na values in the first column with 0
    covid_data.iloc[:, 4:-3].fillna(method='ffill', axis=1, inplace=True)
    restrictions_data = pd.DataFrame()
    restrictions_data['country'] = covid_data['country']
    restrictions_data['last_restrictions'] = covid_data['international_travel_controls']
    restrictions_data = restrictions_data[restrictions_data['last_restrictions'] != -1]

    # dynamically get the world-country boundaries
    res = requests.get(
        "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json")
    countries_shapes = pd.DataFrame(json.loads(res.content.decode()))
    countries_shapes = countries_shapes.assign(id=countries_shapes["features"].apply(pd.Series)["id"],
                                               name=countries_shapes["features"].apply(pd.Series)["properties"].apply(
                                                   pd.Series)["name"])

    # adapt country names to geo data "world-countries.json"
    restrictions_data.country = restrictions_data.country.replace(
        {"US": "United States of America", "Tanzania": "United Republic of Tanzania",
         "Congo (Brazzaville)": "Republic of the Congo", "Congo (Kinshasa)": "Democratic Republic of the Congo",
         "Cote d'Ivoire": "Ivory Coast", "Guinea-Bissau": "Guinea Bissau", "Czechia": "Czech Republic",
         "Serbia": "Republic of Serbia", 'Bahamas': "The Bahamas", "Guinea": "Guinea",
         "Korea, South": 'South Korea', "Taiwan*": "Taiwan", "Timor-Leste": "East Timor"})

    # initialize a folium map
    corona_map = folium.Map(location=[0, 0], zoom_start=3)

    # creating a choropleth map by level of restrictions in differents countries

    def style_fn(feature):
        """
           Define the color of a country by its level of travel restrictions {0, 1, 2, 3, 4}

        """
        country = feature['properties']['name']
        if country in restrictions_data[restrictions_data['last_restrictions'] == 0]['country'].values.tolist():
            style = {'fillColor': 'white', 'color': 'white', 'fillOpacity': 0.8}
            return style
        if country in restrictions_data[restrictions_data['last_restrictions'] == 1]['country'].values.tolist():
            style = {'fillColor': '#ffffcc', 'color': '#00000000', 'fillOpacity': 0.8}
            return style
        if country in restrictions_data[restrictions_data['last_restrictions'] == 2]['country'].values.tolist():
            style = {'fillColor': '#ffeda0', 'color': '#00000000', 'fillOpacity': 0.8}
            return style
        if country in restrictions_data[restrictions_data['last_restrictions'] == 3]['country'].values.tolist():
            style = {'fillColor': '#fd8d3c', 'color': '#00000000', 'fillOpacity': 0.7}
            return style
        if country in restrictions_data[restrictions_data['last_restrictions'] == 4]['country'].values.tolist():
            style = {'fillColor': '#bd0026', 'color': '#00000000', 'fillOpacity': 0.8}
            return style

    # overlay desired countries over folium map with a specific color

    countries = restrictions_data['country'].values.tolist()
    for r in countries_shapes.loc[countries_shapes["name"].isin(countries)].to_dict(orient="records"):
        folium.GeoJson(r["features"], name=r["name"], tooltip=r["name"], style_function=style_fn).add_to(corona_map)
    # Add legend to the map
    template = """
    {% macro html(this, kwargs) %}
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
    </head>
    <body>
      <div id='maplegend' class='maplegend' 
        style='position: absolute; z-index:9999; border:2px solid grey; background-color:rgba(255, 255, 255, 0.8);
         border-radius:6px; padding: 10px; font-size:14px; right: 20px; bottom: 20px;'>

      <div class='legend-title'>International Travel Control</div>
      <div class='legend-scale'>
        <ul class='legend-labels'>
           <li><span style='background:grey;opacity:0.4;'></span>No Data.</li>
           <li><span style='background:white;opacity:0.8;'></span>No measures.</li>
           <li><span style='background:#ffffcc;opacity:0.9;'></span>Screening.</li>
           <li><span style='background:#ffeda0;opacity:0.9;'></span>Quarantine from high-risk regions.</li>
           <li><span style='background:#fd8d3c;opacity:0.8;'></span>Ban on high-risk regions.</li>
           <li><span style='background:#bd0026;opacity:1;'></span>Total border closure.</li>
        </ul>
      </div>
      </div>

    </body>
    </html>

    <style type='text/css'>
      .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
      .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }
      .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }
      .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 1px solid #999;
        }
      .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
        }
      .maplegend a {
        color: #777;
        }
    </style>
    {% endmacro %}"""

    macro = MacroElement()
    macro._template = Template(template)

    corona_map.get_root().add_child(macro)

    # ### Plot number of confirmed cases per country

    for i in range(0, len(covid_data)):
        folium.Circle(
            location=[covid_data.iloc[i]['lat'], covid_data.iloc[i]['long']],
            fill=True,
            radius=(int((np.log(covid_data.iloc[i, -4] + 0.00001))) + 0.2) * 20000,
            color='red',
            fill_color='indigo',
            tooltip="<div style='margin: 0; background-color: black; color: white;'>" +
                    "<h4 style='text-align:center;font-weight: bold'>" + covid_data.iloc[i]['country'] + "</h4>" +
                    "<h4 style='text-align:center;font-weight: bold'>" + str(covid_data.iloc[i]['state']) + "</h4>" +
                    "<hr style='margin:10px;color: white;'>" +
                    "<ul style='color: white;;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>" +
                    "<li>Confirmed cases: " + str(covid_data.iloc[i, -4]) + "</li>" +
                    "</ul></div>",
        ).add_to(corona_map)

    # saving the map as an html file
    covid_map = corona_map.save(outfile="./templates/covid_map.html")

    return render_template('covid_map.html')
