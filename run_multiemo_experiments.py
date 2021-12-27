import logging
import os
import sys

from KD.utils import is_folder_empty

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')
MODEL_FOLDER = os.path.join(PROJECT_FOLDER, 'models')

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

data_dir = os.path.join(DATA_FOLDER, 'multiemo2')

REP_NUM = 4

batch_size = 8
num_train_epochs = 4
learning_rate = 5e-5
weight_decay = 0.01
warmup_steps = 0
max_seq_length = 256

task_name = 'multiemo_en_all_text'


def main():
    print(PROJECT_FOLDER)
    os.chdir(PROJECT_FOLDER)

    if not os.path.exists(os.path.join(DATA_FOLDER, 'multiemo2')):
        logger.info("Downloading Multiemo data")
        cmd = 'python3 scripts/download_dataset.py --data_dir data/multiemo2'
        run_process(cmd)
        logger.info("Downloading finished")

    if not os.path.exists(os.path.join(MODEL_FOLDER, 'bert_pt', 'pytorch_model.bin')):
        logger.info("Downloading bert-base-uncased model")
        cmd = 'python3 download_bert.py'
        run_process(cmd)
        logger.info("Downloading finished")

    os.chdir(os.path.join(PROJECT_FOLDER, 'Pruning'))
    if not os.path.exists(os.path.join(MODEL_FOLDER, 'bert_ft', task_name)):
        cmd = 'python3 -m multiemo_fine_tune_bert '
        options = [
            '--pretrained_model', 'data/models/bert_pt',
            '--data_dir', 'data/multiemo2',
            '--task_name', task_name,
            '--output_dir', f'data/models/bert_ft/{task_name}',
            '--learning_rate', str(learning_rate),
            '--num_train_epochs', str(num_train_epochs),
            '--weight_decay', str(weight_decay),
            '--train_batch_size', str(batch_size),
            '--max_seq_length', str(max_seq_length),
            '--do_lower_case'
        ]
        cmd += ' '.join(options)
        logger.info(f"Fine tuning bert-base-uncased on {task_name}")
        run_process(cmd)

    os.chdir(os.path.join(PROJECT_FOLDER, 'KD'))

    for i in range(REP_NUM):
        bert_student_output_dir = manage_output_dir("../models/bert_student", task_name)
        teacher_model_dir = f'../models/bert_ft/{task_name}'

        cmd = 'python3 -m train '
        options = [
            '--config_dir', 'configurations/config_bert-student.json',
            '--teacher_model', teacher_model_dir,
            '--output_dir', bert_student_output_dir,
            '--task_name', task_name,
            '--train_batch_size', str(batch_size),
            '--max_seq_length', str(max_seq_length),
            '--do_lower_case'
        ]
        cmd += ' '.join(options)
        logger.info(f"Training BERT_BASE (student-intermediary) model on {task_name}")
        run_process(cmd)

        bert_8layer_output_dir = manage_output_dir("../models/bert-8layer/iter_depth_prun", task_name)
        cmd = 'python3 train.py '
        options = [
            '--config_dir', 'configurations/config_bert-8layer.json',
            '--teacher_model', bert_student_output_dir,
            '--student_model', bert_student_output_dir,
            '--output_dir', bert_8layer_output_dir,
            '--task_name', task_name,
            '--train_batch_size', str(batch_size),
            '--max_seq_length', str(max_seq_length),
            '--do_lower_case'
        ]
        cmd += ' '.join(options)
        logger.info(f"Iterative depth pruning to BERT_8 on {task_name}")
        run_process(cmd)

        rosita_output_dir = manage_output_dir("../models/rosita-setting4", task_name)
        cmd = 'python3 -m train '
        options = [
            '--config_dir', 'configurations/config_setting4.json',
            '--teacher_model', bert_8layer_output_dir,
            '--student_model', bert_8layer_output_dir,
            '--output_dir', rosita_output_dir,
            '--task_name', task_name,
            '--train_batch_size', str(batch_size),
            '--max_seq_length', str(max_seq_length),
            '--do_lower_case'
        ]
        cmd += ' '.join(options)
        logger.info(f"Iterative width pruning of BERT_8 on {task_name}")
        run_process(cmd)

        cmd = 'python3 -m test '
        options = [
            '--task_name', task_name,
            '--data_dir', '../data/multiemo2',
            '--student_model', rosita_output_dir,
            '--output_dir', rosita_output_dir,
            '--do_eval',
            '--do_predict',
            '--eval_batch_size', str(batch_size),
            '--max_seq_length', str(max_seq_length),
            '--do_lower_case'
        ]
        cmd += ' '.join(options)
        logger.info(f"Evaluating RoSITA for {task_name}")
        run_process(cmd)

    cmd = f'python3 -m gather_results --task_name {task_name}'
    logger.info(f"Gathering results to csv for {task_name}")
    run_process(cmd)


def run_process(proc):
    os.system(proc)


def manage_output_dir(output_dir: str, task_name: str) -> str:
    output_dir = os.path.join(output_dir, task_name)
    run = 1
    while os.path.exists(output_dir + '-run-' + str(run)):
        if is_folder_empty(output_dir + '-run-' + str(run)):
            logger.info('folder exist but empty, use it as output')
            break
        logger.info(output_dir + '-run-' + str(run) + ' exist, trying next')
        run += 1
    output_dir += '-run-' + str(run)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


if __name__ == '__main__':
    main()
