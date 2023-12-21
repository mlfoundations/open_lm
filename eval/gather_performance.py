import json
from pathlib import Path

def gather_performance(path, model_name):
    with open(path / "groups.json") as f:
        scenarios = json.load(f)
    groups_to_metric = {}
    for scenario in scenarios:
        if scenario["title"] == "Scenarios":
            for group_metadata in scenario["rows"]:
                group_name = group_metadata[0]['value']
                group_shorthand = group_metadata[0]['href'].replace('?group=', '')
                with open(path / "groups" / f'{group_shorthand}.json') as f:
                    metrics = json.load(f)
                for model_metrics in metrics[0]['rows']:
                    if model_metrics[0]['value'] == model_name:
                        break
                preferred_metric_name = metrics[0]['header'][1]['value']
                preferred_metric_value = model_metrics[1]['value']
                dict_name = f'{group_name}_{preferred_metric_name}'
                groups_to_metric[dict_name] = preferred_metric_value
    return groups_to_metric

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    args = parser.parse_args()
    print(gather_performance(Path(args.path), args.model_name))