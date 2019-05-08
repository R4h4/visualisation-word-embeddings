# -*- coding: utf-8 -*
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from app import app, server  # Import server for Gunicorn
from apps import app_w2v_visualized

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Medium", href="https://medium.com/@karsteneckhardt")),
    ],
    brand="Understanding NLP",
    brand_href="#",
    sticky="top",
)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(
        children=[
            navbar,
            dcc.Loading(
                id='page-loader',
                children=html.Div(id='page-content'),
                type="circle",
                fullscreen=True,
            )
        ]
    )
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return app_w2v_visualized.layout
    else:
        return '404'


if __name__ == '__main__':
    app.run_server(debug=True, threaded=True)
