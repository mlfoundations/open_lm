import os
import json
import re
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, ctx


    # Define a consistent symbol map for mult values
symbol_map = {
    '1p0': 'circle',
    '2p0': 'triangle-up',
    '4p0': 'square'
}

def get_value_from_json(file_path, key_list):
    """Extracts the value from a JSON file given a list of nested keys."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        value = data
        for i, key in enumerate(key_list):
            if isinstance(value, dict):
                if key in value:
                    value = value[key]
                else:
                    return None, f"Key '{key}' not found at level {i}. Available keys: {', '.join(value.keys())}"
            else:
                return None, f"Cannot traverse further at key '{key}' (level {i}). Current value is not a dict: {type(value)}"
        
        return value, None
    except json.JSONDecodeError as e:
        return None, f"JSON Decode Error: {str(e)}"
    except IOError as e:
        return None, f"IO Error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"
    
    
def extract_mixing_ratios(filename: str):
    datasets = ['llava', 'dclm', 'datacompdr1b_caption']
    ratios = {dataset: 0 for dataset in datasets}
    
    filename_shortened = filename.replace("llava-multimodal+", "")
    
    is_datacompdr1b = filename_shortened.startswith("datacompdr1b_caption")
    filename_shortened = filename_shortened.replace("datacompdr1b_caption", "")
    while filename_shortened.startswith("_") or filename_shortened.startswith("-"):
        filename_shortened = filename_shortened[1:]
    is_llava = filename_shortened.startswith("llava")
    filename_shortened = filename_shortened.replace("llava", "")
    while filename_shortened.startswith("_") or filename_shortened.startswith("-"):
        filename_shortened = filename_shortened[1:]
    is_dclm = filename_shortened.startswith("dclm")
    filename_shortened = filename_shortened.replace("dclm", "")
    while filename_shortened.startswith("_") or filename_shortened.startswith("-"):
        filename_shortened = filename_shortened[1:]
    
    print(filename_shortened)
    print(f"llava: {is_llava}, dclm: {is_dclm}, datacompdr1b: {is_datacompdr1b}")
        
    if not filename_shortened.startswith("0"):
        if is_datacompdr1b:
            ratios['datacompdr1b_caption'] = 1.0
        if is_llava:
            ratios['llava'] = 1.0
        if is_dclm:
            ratios['dclm'] = 1.0
        print(filename)
        print(ratios)
        return ratios
    
    # find next number in format 0p\d+
    next_number = re.search(r'0p\d+', filename_shortened)
    filename_shortened = filename_shortened[next_number.end():]
    print(filename_shortened)
    if is_datacompdr1b:
        ratios['datacompdr1b_caption'] = float(next_number.group().replace('p', '.'))
    elif is_llava:
        is_llava = False
        ratios['llava'] = float(next_number.group().replace('p', '.'))
        
    next_number = re.search(r'0p\d+', filename_shortened)
    filename_shortened = filename_shortened[next_number.end():]
    print(filename_shortened)
    if is_llava:
        ratios['llava'] = float(next_number.group().replace('p', '.')) 
    elif is_dclm:
        ratios['dclm'] = float(next_number.group().replace('p', '.'))
        print(filename)
        print(ratios)  
        return ratios
    
    if is_dclm:
        next_number = re.search(r'0p\d+', filename_shortened)
        ratios['dclm'] = float(next_number.group().replace('p', '.'))
    
    print(filename)
    print(ratios)    
    return ratios


def extract_epoch_from_name(dirname):
    """Extracts the epoch number from the directory name (assumed format 'epochs=<number>')."""
    match = re.search(r'epochs=(\d+)', dirname)
    if match:
        epoch = int(match.group(1))
        return epoch
    return None


def extract_mult_from_name(dirname):
    """Extracts the mult value from the directory name (assumed format 'mult=<value>')."""
    match = re.search(r'mult=(\d+p\d+)', dirname)
    if match:
        return match.group(1)
    return None

# Update the extract_values function to use the new return value of get_value_from_json
def extract_values(base_dir_1, base_dir_2, key_list_1, key_list_2):
    data = []
    errors = []

    try:
        files_in_base_dir_1 = os.listdir(base_dir_1)
    except Exception as e:
        errors.append(f"Error reading directory {base_dir_1}: {str(e)}")
        return pd.DataFrame(), errors

    for base_name in files_in_base_dir_1:
        try:
            file_path = os.path.join(base_dir_1, base_name)
            base_name = base_name.split(".")[0]
            
            x_value, x_error = get_value_from_json(file_path, key_list_1)
            if x_error:
                errors.append(f"Error extracting x_value from {file_path}: {x_error}")
                continue
            
            aggregated_file = os.path.join(base_dir_2, f'{base_name}.json')
            if not os.path.exists(aggregated_file):
                errors.append(f"Aggregated file not found: {aggregated_file}")
                continue
            
            y_value, y_error = get_value_from_json(aggregated_file, key_list_2)
            if y_error:
                errors.append(f"Error extracting y_value from {aggregated_file}: {y_error}")
                continue
            
            epoch = extract_epoch_from_name(base_name)
            mult = extract_mult_from_name(base_name)
            
            if epoch is None or mult is None:
                errors.append(f"Unable to extract epoch or mult from {base_name}")
                continue
            
            data.append({
                'x': x_value,
                'y': y_value,
                'epoch': epoch,
                'mult': mult,
                'file_name': base_name
            })
        except Exception as e:
            errors.append(f"Error processing {base_name}: {str(e)}")

    return pd.DataFrame(data), errors
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Interactive Data Explorer"),
    
    html.Div([
        html.Label("Bucket:"),
        dcc.Input(id='bucket-input', type='text', value='1b'),
        
        html.Label("Base Directory 1:"),
        dcc.Input(id='base-dir-1-input', type='text', disabled=True),  # Automated, disabled for manual input
        
        html.Label("Base Directory 2:"),
        dcc.Input(id='base-dir-2-input', type='text', disabled=True),  # Automated, disabled for manual input
        
        html.Label("Key List 1:"),
        dcc.Dropdown(
            id='key-list-1-dropdown',
            options=[
                {'label': 'agieval_sat_en', 'value': 'results/agieval_sat_en/acc,none'},
                {'label': 'arc_easy', 'value': 'results/arc_easy/acc,none'},
                {'label': 'bigbench_conceptual_combinations', 'value': 'results/bigbench_conceptual_combinations_multiple_choice/acc,none'},
                {'label': 'bigbench_cs_algorithms', 'value': 'results/bigbench_cs_algorithms_multiple_choice/acc,none'},
                {'label': 'boolq', 'value': 'results/boolq/acc,none'},
                {'label': 'copa', 'value': 'results/copa/acc,none'},
                {'label': 'hellaswag', 'value': 'results/hellaswag/acc,none'},
                {'label': 'mathqa', 'value': 'results/mathqa/acc,none'},
                {'label': 'piqa', 'value': 'results/piqa/acc,none'},
                {'label': 'pubmedqa', 'value': 'results/pubmedqa/acc,none'},
                {"label": "z-score", "value": "z-score/value"}
            ],
            value='z-score/value'  # Default value
        ),
        
        html.Label("Key List 2:"),
        dcc.Dropdown(
            id='key-list-2-dropdown',
            options=[
                {'label': 'pope', 'value': 'pope_pope-full/accuracy__POPE-final-Accuracy'},
                {'label': 'OCIDRe', 'value': 'ocid-ref_ocid-ref-full/accuracy__OCIDRef-All'},
                {'label': 'RefCOCO', 'value': 'refcoco_refcoco-full/accuracy__RefCOCO'},
                {'label': 'TextVQA', 'value': 'text-vqa_text-vqa-full/accuracy__TextVQA-Pure'},
                {'label': 'VizWiz', 'value': 'vizwiz_vizwiz-full/accuracy__VizWiz-Overall'},
                {'label': 'gqa', 'value': 'gqa_gqa-full/accuracy'},
                {'label': 'vqa-v2', 'value': 'vqa-v2_vqa-v2-full/accuracy'},
                {"label": "z-score", "value": "z-score/value"}
            ],
            value='z-score/value'  # Default value
        ),
        
        html.Button('Update Plot', id='update-button', n_clicks=0)
    ]),
    
    dcc.Graph(id='scatter-plot'),
    
    html.Label("Mixing Ratios:"),
    html.Label("Llava Ratio Range:"),
    dcc.RangeSlider(
        id='llava-slider', 
        min=0, max=100, step=1, value=[0, 100],
        marks={i: f'{i}%' for i in range(0, 101, 20)}
    ),
    
    html.Label("DCLM Ratio Range:"),
    dcc.RangeSlider(
        id='dclm-slider', 
        min=0, max=100, step=1, value=[0, 100],
        marks={i: f'{i}%' for i in range(0, 101, 20)}
    ),
    
    html.Label("Datacompdr1b Ratio Range:"),
    dcc.RangeSlider(
        id='datacompdr1b-slider', 
        min=0, max=100, step=1, value=[0, 100],
        marks={i: f'{i}%' for i in range(0, 101, 20)}
    ),
    
    html.Div([
        dcc.Dropdown(id='epoch-dropdown', placeholder="Select Epoch"),
        dcc.Dropdown(id='mult-dropdown', placeholder="Select Mult")
    ]),
    
    html.Div(id='error-output'),
    html.Div(id='debug-output')
])

@app.callback(
    Output('base-dir-1-input', 'value'),
    Output('base-dir-2-input', 'value'),
    Input('bucket-input', 'value')
)
def update_base_dirs(bucket):
    """Automatically set base_dir_1 and base_dir_2 based on bucket input."""
    if not bucket:
        base_dir_1 = "results/mbm_paper_texteval"
        base_dir_2 = "results/mbm_paper_eval/aggregated"
    else:
        base_dir_1 = f"results/mbm_paper_texteval_{bucket}"
        base_dir_2 = f"results/mbm_paper_eval_{bucket}/aggregated"
    
    return base_dir_1, base_dir_2

def create_interactive_plot(df, x_label, y_label):
    if df.empty:
        return px.scatter(title="No data to display")
    
    fig = px.scatter(df, x='x', y='y', color='epoch', symbol='mult', symbol_map=symbol_map, hover_data=['file_name', 'epoch', 'mult'],
                     labels={'x': x_label, 'y': y_label, 'epoch': 'Epoch', 'mult': 'Mult'},
                     title=f"{x_label} vs {y_label}")

    fig.update_traces(marker=dict(size=10))
    fig.update_layout(legend_title_text='Epoch')

    return fig

def is_in_range(value, range_values):
    if value is None:
        return True
    return range_values[0] <= value <= range_values[1]


@app.callback(
    Output('scatter-plot', 'figure'),
    Output('epoch-dropdown', 'options'),
    Output('mult-dropdown', 'options'),
    Output('error-output', 'children'),
    Output('debug-output', 'children'),
    Input('update-button', 'n_clicks'),
    Input('epoch-dropdown', 'value'),
    Input('mult-dropdown', 'value'),
    State('bucket-input', 'value'),
    State('base-dir-1-input', 'value'),
    State('base-dir-2-input', 'value'),
    State('key-list-1-dropdown', 'value'),
    State('key-list-2-dropdown', 'value'),
    State('llava-slider', 'value'),
    State('dclm-slider', 'value'),
    State('datacompdr1b-slider', 'value'),
    allow_duplicate=True
)
def update_plot(n_clicks, selected_epoch, selected_mult, bucket, base_dir_1, base_dir_2, key_list_1, key_list_2, llava_range, dclm_range, datacompdr1b_range):
    # Same as before, just use the dropdown values for key_list_1 and key_list_2
    key_list_1 = key_list_1.split('/')
    key_list_2 = key_list_2.split('/')
    
    df, errors = extract_values(base_dir_1, base_dir_2, key_list_1, key_list_2)
    
    debug_info = f"Processed {len(df)} valid entries. Encountered {len(errors)} errors."
    
    if df.empty:
        return px.scatter(title="No data to display"), [], [], html.Ul([html.Li(error) for error in errors]), debug_info
    
    x_label = f"{key_list_1[0]} {key_list_1[1]}"
    y_label = f"{key_list_2[0]} {key_list_2[1]}"
    
    fig = create_interactive_plot(df, x_label, y_label)
    
    try:
        # Filter based on the mixing ratios
        df['mixing_ratios'] = df['file_name'].apply(extract_mixing_ratios)
    
        df = df[
            (df['mixing_ratios'].apply(lambda x: is_in_range(x.get("llava", None), llava_range))) &
            (df['mixing_ratios'].apply(lambda x: is_in_range(x.get("dclm", None), dclm_range))) &
            (df['mixing_ratios'].apply(lambda x: is_in_range(x.get("datacompdr1b_caption", None), datacompdr1b_range)))
        ]
    except Exception as e:
        print(f"Error filtering by mixing ratios: {str(e)}")
        
    epoch_options = [{'label': f'Epoch {e}', 'value': e} for e in sorted(df['epoch'].unique())]
    epoch_options.insert(0, {'label': 'All Epochs', 'value': 'all'})
    
    mult_options = [{'label': f'Mult {m}', 'value': m} for m in sorted(df['mult'].unique())]
    mult_options.insert(0, {'label': 'All Mults', 'value': 'all'})

    if selected_epoch and selected_epoch != 'all':
        df = df[df['epoch'] == int(selected_epoch)]
    
    if selected_mult and selected_mult != 'all':
        df = df[df['mult'] == selected_mult]
    
    if df.empty:
        return px.scatter(title="No data to display after filtering")
    
    fig = px.scatter(df, x='x', y='y', color='epoch', symbol='mult', symbol_map=symbol_map, hover_data=['file_name', 'epoch', 'mult'],
                     labels=fig['layout']['xaxis']['title'],
                     title=fig['layout']['title']['text'])
    
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(legend_title_text='Epoch')
    
    error_output = html.Ul([html.Li(error) for error in errors]) if errors else "No errors"
    
    return fig, epoch_options, mult_options, error_output, debug_info

# ... (rest of the script remains the same) ...

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8051)