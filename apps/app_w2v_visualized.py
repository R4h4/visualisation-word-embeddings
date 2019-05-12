# -*- coding: utf-8 -*-
import logging

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
# from flask_caching import Cache

import numpy as np
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE

from app import app

app.title = 'Word-Embeddings Visualized by Karsten Eckhardt'

# TODO: Implement Caching of the Dataset + Dropdown
# cache = Cache(app.server, config={
#    'CACHE_TYPE': 'filesystem',
#    'CACHE_DIR': 'cache-directory'
#})

#TIMEOUT = 60

"""
@cache.memoize(timeout=TIMEOUT)
def load_model():
    # Shout-out to GitHub user eyaler for creating a small version of word2vec (https://github.com/eyaler/word2vec-slim)
    model = KeyedVectors.load_word2vec_format('https://s3-ap-southeast-1.amazonaws.com/understanding-nlp/word2vec.twitter.27B.25d.bin', binary=True)
    return model
"""


MODEL = KeyedVectors.load_word2vec_format('https://s3-ap-southeast-1.amazonaws.com/understanding-nlp/word2vec.twitter.27B.25d.bin', binary=True)

layout = dbc.Container(
    children=[
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("Word-embeddings"),
                        html.P(
                            """\
                            At its core, word-embeddings are a learned lookup-table that assign a vector to each word. \
                            What makes them so interesting for NLP is that words which are used in a similar context \
                            and/or have a similar meaning, are grouped together in the vector-space. Generally, those \
                            vectors have ~300 dimensions. However, t-SNE (t-distributed stochastic neighbor embedding) \
                            allows to get an impression of word relationships on an humanly comprehensible 2d field. 
                            """
                        ),
                        html.P(
                            """\
                            To get a better intuition, simply pick one or more words from the drop-down. If more than \
                            one word is picked, every word-cloud is separated by color. Use pinch-to-zoom or draw a \
                            rectangle to get a closer look.
                            """
                        )
                    ],
                    md=4,
                ),

                dbc.Col(
                    [
                        dcc.Dropdown(
                            id='words',
                            style={
                                'margin-left': '6rm'
                            },
                            clearable=False,
                            options= [{'label': word, 'value': word} for word in ['food', 'coffee', 'sex', 'viagra',
                                                                                  'example', 'king', 'male']], #[{'label': word, 'value': word} for word in MODEL.index2word],
                            value='example',
                            multi=True
                        ),

                        dcc.Loading(
                            id="loading-2",
                            children=[
                                dcc.Graph(
                                    id='word-embeddings',
                                    className='demo-layout-container',
                                    # Initialize the graph with the right axis
                                    figure={
                                        'data': ['example'],
                                        'layout': {
                                            'xaxis': dict(
                                                    showgrid=False,
                                                    zeroline=False,
                                                    showticklabels=False
                                                ),
                                            'yaxis': dict(
                                                    showgrid=False,
                                                    zeroline=False,
                                                    showticklabels=False
                                                ),
                                        }
                                    }
                                ),
                            ],
                            type="circle",
                        ),
                    ],

                )
            ],
        ),
    ],
    className="mt-4",
)


def tsne_plot_similar_words(labels, embedding_clusters, word_clusters):
    colors = [i for i in range(len(labels))]  # cm.rainbow(np.linspace(0, 1, len(labels)))
    return [go.Scatter(
                x=embeddings[:, 0],
                y=embeddings[:, 1],
                name=label,
                mode='markers+text',
                text=words,
                textposition='bottom center',
                marker=dict(
                    size=10,
                    color=color,
                )
            ) for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors)]


@app.callback(Output('word-embeddings', 'figure'),
              [Input('words', 'value')])
def callback(picked_words):
    logging.debug(f"Reloading chart, picked words are: {picked_words}.")

    axis_style = dict()
    _ = dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False
    )

    if picked_words:
        embedding_clusters = []
        word_clusters = []
        for word in picked_words:
            embeddings = [MODEL[word]]
            words = [word]
            for similar_word, _ in MODEL.wv.most_similar(word, topn=20):
                words.append(similar_word)
                embeddings.append(MODEL[similar_word])
            embedding_clusters.append(embeddings)
            word_clusters.append(words)

        embedding_clusters = np.array(embedding_clusters)
        n, m, k = embedding_clusters.shape
        tsne_model_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
        embeddings_2d = np.array(tsne_model_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m,
                                                                                                                  2)
        data = tsne_plot_similar_words(picked_words, embeddings_2d, word_clusters)

    else:
        logging.info("No words chosen, return empty grid.")
        data = []

    return {'data': data,
                'layout': {
                    'opacity': '0.8',
                    'xaxis':axis_style,
                    'yaxis': axis_style,
                    'gradient': {
                        'type': 'radial'
                    }
                }
            }

