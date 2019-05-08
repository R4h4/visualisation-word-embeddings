# -*- coding: utf-8 -*-
import dash
import dash_bootstrap_components as dbc
import os
from random import randint

external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True
server = app.server
server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))
