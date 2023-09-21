from dash import Input, Output, dash , dcc, html ,State
import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
import dash
import dash.dependencies as dd
import base64
import networkx as nx
import base64
import os
import pickle
import pandas as pd


pickle_files = ['df_device.pkl', 'traffic_YH.pkl', 'click.pkl', 'average_diff.pkl', 'cust_trx5.pkl', 'search_count.pkl', 'df_weights.pkl']
dataframes = {}

for pickle_file in pickle_files:
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)
        df = pd.DataFrame(data)
        # Use the file name (without extension) as the dataframe key
        dataframe_name = pickle_file.split('.')[0]
        dataframes[dataframe_name] = df
df_device = dataframes['df_device']
traffic_YH = dataframes['traffic_YH']
click = dataframes['click']
average_diff = dataframes['average_diff']
cust_trx5 = dataframes['cust_trx5']
search_count = dataframes['search_count']
df_weights = dataframes['df_weights']


app = dash.Dash(__name__, suppress_callback_exceptions=True)
# Function to generate the traffic page layout
def generate_traffic_page():
    return html.Div([
        html.H1("Stacked Bar Chart for Traffic Events"),

        dcc.Dropdown(
            id='year-dropdown',
            options=[{'label': year, 'value': year} for year in traffic_YH['year'].unique()],
            value=traffic_YH['year'].max(),
            multi=False
        ),

        dcc.Graph(id='stacked-bar-chart')
    ])
# Function to generate the traffic page layout
def generate_traffic_source():
    return html.Div([
        html.H1("Pie Chart for Traffic Sources"),
        dcc.Graph(id='pie-chart-traffic')
    ])
    
# Define the click page layout
def generate_click_page():
    return html.Div([
        html.H1("Pie Chart for Event Percentage (Click)"),

        dcc.Dropdown(
            id='year-dropdown-click',
            options=[{'label': year, 'value': year} for year in click['Year'].unique()],
            multi=True,
            value=[2022]  # Default selection
        ),

        dcc.Graph(id='pie-chart-click')
    ])

# Callback to update the pie chart (Click page) based on selected years
@app.callback(
    Output('pie-chart-click', 'figure'),
    Input('year-dropdown-click', 'value')
)
def update_pie_chart_click(selected_years):
    # Filter the DataFrame based on selected years
    filtered_df = click[click['Year'].isin(selected_years)]

    # Sum the percentages for the selected years
    event_totals = filtered_df.groupby('Event')['Percentage'].sum().reset_index()

    # Create the Pie Chart
    fig = px.pie(event_totals, names='Event', values='Percentage', title='Event Percentage Distribution (Click)')

    return fig

# Callback to update the stacked bar chart based on the selected year
@app.callback(
    Output('stacked-bar-chart', 'figure'),
    Input('year-dropdown', 'value')
)
def update_stacked_bar_chart(selected_year):
    filtered_df = traffic_YH[traffic_YH['year'] == selected_year]

    fig = px.bar(filtered_df, x='Hour', y=['all', 'booking'],
                 title=f"Stacked Bar Chart for Traffic Events in {selected_year}",
                 labels={'Hour': 'Hour', 'value': 'Count'},
                 height=400)

    fig.update_layout(barmode='stack')

    return fig
# Callback to update the traffic pie chart
@app.callback(
    Output('pie-chart-traffic', 'figure'),
    Input('pie-chart-traffic', 'relayoutData')
)
def update_pie_chart_traffic(relayout_data):
    fig = px.pie(df_device, values='traffic_source', names='index', title='Traffic Sources')
    return fig
def generate_churn_page():
    return html.Div([
        html.H1("Churn Page"),
        dcc.Tabs(id="tabs", value='tab-1', children=[
            dcc.Tab(label='Sankey Diagram', value='tab-1'),
            dcc.Tab(label='Heatmap', value='tab-2'),
            dcc.Tab(label='Churn Rate Over Time', value='tab-3'),
            dcc.Tab(label='Churn Distribution Histogram', value='tab-4'),
        ]),
        html.Div(id='tab-content'),
        # Add the RangeSlider for x-axis range selection
        dcc.RangeSlider(
            id='x-axis-range-slider',
            min=0,
            max=50,
            step=0.1,
            value=[0, 15],
            marks={0: '0', 50: '50'},
        ),
    ])
# Callback to render the selected tab content
@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H1("Customer Cohort Sankey Diagram"),
            dcc.Graph(id='sankey-diagram')
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H1("Customer Churn Visualization"),
            dcc.Dropdown(
                id='cohort-dropdown',
                options=[
                    {'label': 'All Customers', 'value': 'all'},
                    {'label': 'Churned Customers', 'value': 'churned'},
                    {'label': 'Not Churned Customers', 'value': 'not_churned'}
                ],
                value='all',  # Default selection
                multi=False
            ),
            dcc.Graph(id='heatmap')
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H1("Monthly Churn Rate Over Time"),
            dcc.Graph(id='churn-rate-over-time')
        ])
    elif tab == 'tab-4':
        return html.Div([
            html.H1("Churn Distribution Histogram"),
            dcc.Graph(id='churn-histogram')
        ])
            
G = nx.DiGraph()
for index, row in df_weights.iterrows():
    source_node, target_node = row['Event Transition']
    weight = row['Weight']
    G.add_edge(source_node, target_node, weight=weight)


def update_image_src():
    image_path = os.path.join(base_dir, './Picture1.png')  

    # Encode the image file to base64
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    return f'data:image/png;base64,{encoded_image}'  # Return the encoded image data URI

def generate_image_src2():
    image_path = os.path.join(base_dir, './Picture1.png')  

    # Encode the image file to base64
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    return f'data:image/png;base64,{encoded_image}'

def generate_predict_page():
    return html.Div([
        html.H1("Prediction Page"),
        html.Div([
            html.Img(src=generate_image_src2(), style={'width': '80%'}),
        ], style={'text-align': 'center'}),
    ])
base_dir = os.getcwd()


@app.callback(
    Output('generate_predict_page', 'children'), 
    Input('generate_predict_page', 'relayoutData')
)

def update_predict_page(relayout_data):
    return generate_predict_page()

def generate_weight_click_page():
    return html.Div([
        html.H1("Click stream Page"),
        dcc.Tabs(id="tabs2", value='update_image_src', children=[
            dcc.Tab(label='Directed Graph', value='graph'),
            dcc.Tab(label='Heatmap Matrix', value='heatmap'),
            dcc.Tab(label='Weight of Click', value='update_image_src')  # Corrected value
        ]),
        html.Div(id='tab-content2'),
    ])
    
@app.callback(
    dd.Output('tab-content2', 'children'),
    [dd.Input('tabs2', 'value')]
)
def render_content2(tab):
    if tab == 'graph':
        edge_x = []
        edge_y = []
        edge_text = []
        for edge in G.edges():
            x0, y0 = edge[0], edge[1]
            edge_x.append(x0)
            edge_y.append(y0)
            edge_text.append(f"Weight: {G[x0][y0]['weight']}")

        graph_figure = {
            'data': [go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines+markers',
                line=dict(width=1),
                marker=dict(size=10, opacity=0.5),
                    text=edge_text,
                hoverinfo='text',
                name='Edges',
            )],
            'layout': go.Layout(
                title="Directed Graph",
                showlegend=True,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False),
            )
        }
        return dcc.Graph(figure=graph_figure)
    elif tab == 'heatmap':
        heatmap_figure = px.imshow(pd.crosstab(df_weights['Event Transition'].str[0], df_weights['Event Transition'].str[1], values=df_weights['Weight'], aggfunc='sum'))
        heatmap_figure.update_layout(title="Heatmap Matrix")
        return dcc.Graph(figure=heatmap_figure)
    elif tab == 'update_image_src':
        return html.Div([
            html.H1("Weight of Click Page"),
            html.Div(
        html.Img(src=update_image_src()),
        style={'display': 'flex', 'justify-content': 'center', 'margin': '20px'}
    )
        ])


# Callback to update the Sankey diagram based on selected cohorts
@app.callback(
    Output('sankey-diagram', 'figure'),
    Input('sankey-diagram', 'relayoutData')
)
def update_sankey_diagram(relayout_data):
    average_diff['cohort_index'] = average_diff['cohort'].factorize()[0] 
    average_diff['churn_status_index'] = average_diff['churn_status'].factorize()[0]
    # Group data by cohort transitions
    cohort_transitions = average_diff.groupby(['cohort_index', 'churn_status_index']).size().reset_index(name='count')

    # Create a Sankey diagram trace
    trace = go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=list(pd.unique(average_diff[['cohort', 'churn_status']].values.ravel('K')))
        ),
        link=dict(
            source=cohort_transitions['cohort_index'].tolist(),
            target=[len(pd.unique(average_diff['cohort_index'])) + x for x in cohort_transitions['churn_status_index'].tolist()],
            value=cohort_transitions['count'].tolist()
        )
    )

    layout = dict(
        title="Customer Cohort Sankey Diagram",
        font=dict(size=10)
    )

    fig = dict(data=[trace], layout=layout)
    return fig

# Callback to update the heatmap based on selected cohort
@app.callback(
    Output('heatmap', 'figure'),
    Input('cohort-dropdown', 'value')
)
def update_heatmap(selected_cohort):
    if selected_cohort == 'all':
        filtered_df = average_diff
    elif selected_cohort == 'churned':
        filtered_df = average_diff[average_diff['churn_status'] == 'Churned']
    elif selected_cohort == 'not_churned':
        filtered_df = average_diff[average_diff['churn_status'] == 'Not Churned']
    
    # Pivot the DataFrame for the heatmap
    heatmap_data = filtered_df.pivot(index='customer_id', columns='created_at', values='diff')
    
    # Create the Heatmap
    fig = px.imshow(heatmap_data, x=heatmap_data.columns, y=heatmap_data.index, labels=dict(x="Month", y="Customer ID"))
    
    return fig
# Callback to update the churn rate over time graph
@app.callback(
    Output('churn-rate-over-time', 'figure'),
    Input('churn-rate-over-time', 'relayoutData')
)

def update_churn_rate_over_time(relayout_data):
    if relayout_data is not None and 'xaxis.range' in relayout_data:
        x_range = relayout_data['xaxis.range']
        start_date = pd.to_datetime(x_range[0])
        end_date = pd.to_datetime(x_range[1])
        filtered_average_diff = average_diff[
            (average_diff['created_at'] >= start_date) & (average_diff['created_at'] <= end_date)
        ]
    else:
        filtered_average_diff = average_diff[average_diff['created_at'] >= '2021-09-01']

    # Group the data by month and churn status
    churn_data = filtered_average_diff.groupby([pd.Grouper(key='created_at', freq='M'), 'churn_status']).size().unstack(fill_value=0)

    # Calculate churn rate (Churned / Total Customers) for each month
    churn_data['Total Customers'] = churn_data['Churned'] + churn_data['Not Churned']
    churn_data['Churn Rate'] = churn_data['Churned'] / churn_data['Total Customers'] * 100

    # Create a line chart for churn rate over time
    fig = px.line(churn_data, x=churn_data.index, y='Churn Rate', title='Monthly Churn Rate Over Time')
    
    return fig


# Callback to update the churn distribution histogram
@app.callback(
    Output('churn-histogram', 'figure'),
    Input('churn-histogram', 'relayoutData'),
    State('x-axis-range-slider', 'value')  # Get the selected x-axis range
)
def update_churn_histogram(relayout_data, x_axis_range):
    fig = px.histogram(cust_trx5, x='diff', title='Month Difference Distribution') 
    fig.update_xaxes(range=x_axis_range)
    
    return fig

# Function to generate the home page layout with navigation links to other pages
def generate_home_page():
    return html.Div([
        html.H1("Select a Page"),
        # dcc.Link('Traffic Page', href='/traffic'),
        html.Br(),
        dcc.Link('traffic Page', href='/traffic'),
        html.Br(),
        dcc.Link('Click Page', href='/click'),  
        html.Br(),
        dcc.Link('Search Page', href='/search'),
        html.Br(),
        dcc.Link('Churn Page', href='/churn') ,
        html.Br(),
        dcc.Link('click stream Page', href='/weight-of-click'),
        html.Br(),
        dcc.Link('prediction Page', href='/predict')

    ])

# Callback to switch between pages
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    # if pathname == '/traffic':
    #     return generate_traffic_page()
    if pathname == '/search':
        return generate_search_page()
    elif pathname == '/traffic':
        return generate_combined_page()
    elif pathname == '/click':  
        return generate_click_page()
    elif pathname == '/churn':  
        return generate_churn_page()
    elif pathname == '/weight-of-click':  
        return generate_weight_click_page()
    elif pathname == '/predict':  
        return generate_predict_page()
    else:
        return generate_home_page()

# Function to generate the search page layout
def generate_search_page():
    return html.Div([
        html.H1("Stacked Bar Chart for Search Counts"),
        dcc.Dropdown(
            id='count-selection',
            options=[
                {'label': 'Count', 'value': 'count'},
                {'label': 'Count All', 'value': 'count_all'},
                {'label': 'Count Without', 'value': 'count_without'}
            ],
            value=['count', 'count_all', 'count_without'],
            multi=True
        ),
        dcc.Graph(id='stacked-bar-chart-search')
    ])

# Callback to update the search stacked bar chart
@app.callback(
    Output('stacked-bar-chart-search', 'figure'),
    Input('count-selection', 'value')
)
def update_stacked_bar_chart_search(selected_counts):
    selected_df = search_count[['search_keywords'] + selected_counts]
    fig = px.bar(selected_df, x='search_keywords', y=selected_counts,
                 title="Stacked Bar Chart for Search Counts",
                 labels={'search_keywords': 'Search Keywords', 'value': 'Count'},
                 category_orders={"search_keywords": selected_df["search_keywords"].tolist()},
                 height=400)
    fig.update_layout(barmode='stack')
    return fig

def generate_combined_page():
    return html.Div([
        html.H1("Combined Page"),
        html.P("This is the combined page that includes elements from both the Traffic and source of it."),
        generate_traffic_page(),  
        generate_traffic_source()    
    ])

# Define the app layout with dcc.Location
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)