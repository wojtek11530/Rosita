import argparse
import json
import os
from typing import Any, Dict

import pandas as pd

from test import MultiemoProcessor
from transformer.modeling_prun import TinyBertForSequenceClassification as PrunBertForSequenceClassification

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')
MODELS_FOLDER = os.path.join(PROJECT_FOLDER, 'models')
FT_MODELS_FOLDER = os.path.join(MODELS_FOLDER, 'setting4')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")

    args = parser.parse_args()
    task_name = args.task_name

    models_subdirectories = get_immediate_subdirectories(FT_MODELS_FOLDER)
    print(FT_MODELS_FOLDER)
    print(models_subdirectories)

    data = list()
    for subdirectory in models_subdirectories:
        data_dict = gather_results(subdirectory, task_name)
        data.append(data_dict)

    df = pd.DataFrame(data)
    cols = df.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    df = df[cols]
    df.to_csv(os.path.join(DATA_FOLDER, 'results-rosita-' + task_name + '.csv'), index=False)


def get_immediate_subdirectories(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def gather_results(model_dir: str, task_name: str) -> Dict[str, Any]:
    task_subfolder = os.path.basename(model_dir)

    with open(os.path.join(model_dir, 'training_params.json')) as json_file:
        training_data_dict = json.load(json_file)

    with open(os.path.join(model_dir, 'test_results.json')) as json_file:
        test_data = json.load(json_file)
        [test_data_dict] = pd.json_normalize(test_data, sep='_').to_dict(orient='records')

    data = training_data_dict.copy()  # start with keys and values of x
    data.update(test_data_dict)

    with open(os.path.join(MODELS_FOLDER, 'bert_student', task_subfolder, 'training_params.json')) as json_file:
        bert_student_training_data_dict = json.load(json_file)
        data['training_time'] = data['training_time'] + bert_student_training_data_dict['training_time']

    with open(os.path.join(MODELS_FOLDER, 'bert-8layer', 'iter_depth_prun', task_subfolder,
                           'training_params.json')) as json_file:
        bert_8layer_training_data_dict = json.load(json_file)
        data['training_time'] = data['training_time'] + bert_8layer_training_data_dict['training_time']

    model_size = os.path.getsize(os.path.join(model_dir, 'pytorch_model.bin'))
    data['model_size'] = model_size

    if 'multiemo' not in task_name:
        raise ValueError("Task not found: %s" % task_name)

    _, lang, domain, kind = task_name.split('_')
    processor = MultiemoProcessor(lang, domain, kind)
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # LOADING THE BEST MODEL
    model = PrunBertForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)

    memory_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    memory_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    memory_used = memory_params + memory_buffers  # in bytes

    data['memory'] = memory_used

    parameters_num = 0
    for n, p in model.named_parameters():
        parameters_num += p.nelement()

    data['parameters'] = parameters_num
    data['name'] = os.path.basename(model_dir)
    data['model_name'] = 'Rosita'
    print(data)
    return data


if __name__ == '__main__':
    main()
