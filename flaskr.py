import pandas as pd
import numpy as np
import re
from flask import Flask, request, render_template


app = Flask(__name__) # create the application instance :)
#app.config.from_object(__name__) # load config from this file , flaskr.py

outlets = pd.read_csv("outlets.csv")
outlet_list = outlets.shop.tolist()
url_front = 'https://www.yelp.com/biz/'
outlet_url = [url_front+x for x in outlet_list]
url_alias = list(zip(outlet_url,outlet_list))

@app.route('/')

def initial_ratings():
    return render_template('initial_ratings.html', list=outlet_list, url_alias=url_alias)

#def show_entries():
#    db = get_db()
#    cur = db.execute('select title, text from entries order by id desc')
#    entries = cur.fetchall()
#    return render_template('show_entries.html', entries=entries)

@app.route('/', methods=["GET", "POST"])
