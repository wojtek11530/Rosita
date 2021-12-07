import logging
import os
import sys

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')
MODEL_FOLDER = os.path.join(PROJECT_FOLDER, 'models')

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

data_dir = os.path.join(DATA_FOLDER, 'multiemo2')

batch_size = 16
num_train_epochs = 3
learning_rate = 5e-5
weight_decay = 0.01


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

    if not os.path.exists(os.path.join(MODEL_FOLDER, 'bert_ft', 'multiemo_en_all_sentence')):
        cmd = 'python3 -m Pruning.multiemo_fine_tune_bert '
        options = [
            '--pretrained_model', 'data/models/bert_pt',
            '--data_dir', 'data/multiemo2',
            '--task_name', 'multiemo_en_all_sentence',
            '--output_dir', 'data/models/bert_ft/multiemo_en_all_sentence',
            '--learning_rate', str(learning_rate),
            '--num_train_epochs', str(num_train_epochs),
            '--weight_decay', str(weight_decay),
            '--train_batch_size', str(batch_size),
            '--do_lower_case'
        ]
        cmd += ' '.join(options)
        logger.info("Fine tuning bert-base-uncased on multiemo_en_all_sentence")
        run_process(cmd)

    if not os.path.exists(os.path.join(MODEL_FOLDER, 'bert_student', 'multiemo_en_all_sentence')):
        cmd = 'python3 -m KD.train '
        options = [
            '--config_dir', 'KD/configurations/config_bert-student.json',
            '--task_name', 'multiemo_en_all_sentence',
            '--do_lower_case'
        ]
        cmd += ' '.join(options)
        logger.info("Training BERT_BASE (student-intermediary) model on multiemo_en_all_sentence")
        run_process(cmd)

    if not os.path.exists(os.path.join(MODEL_FOLDER, 'bert-8layer', 'iter_depth_prun', 'multiemo_en_all_sentence')):
        cmd = 'python3 KD/train.py '
        options = [
            '--config_dir', 'KD/configurations/config_bert-8layer.json',
            '--task_name', 'multiemo_en_all_sentence',
            '--do_lower_case'
        ]
        cmd += ' '.join(options)
        logger.info("Iterative depth pruning to BERT_8 on multiemo_en_all_sentence")
        run_process(cmd)

    if not os.path.exists(os.path.join(MODEL_FOLDER, 'setting4', 'multiemo_en_all_sentence')):
        cmd = 'python3 -m KD.train '
        options = [
            '--config_dir', 'KD/configurations/config_setting4.json',
            '--task_name', 'multiemo_en_all_sentence',
            '--do_lower_case'
        ]
        cmd += ' '.join(options)
        logger.info("Iterative width pruning of BERT_8 on multiemo_en_all_sentence")
        run_process(cmd)

    cmd = 'python3 -m KD.test '
    options = [
        '--task_name', 'multiemo_en_all_sentence',
        '--data_dir', 'data/multiemo2',
        '--student_model', 'models/setting4/multiemo_en_all_sentence',
        '--output_dir', 'models/setting4/multiemo_en_all_sentence',
        '--do_eval',
        '--do_predict'
    ]
    cmd += ' '.join(options)
    logger.info(f"Evaluating RoSITA for multiemo_en_all_sentence")
    run_process(cmd)

    # cmd = f'python3 -m gather_results --task_name multiemo_en_all_sentence'
    # logger.info(f"Gathering results to csv for multiemo_en_all_sentence")
    # run_process(cmd)


def run_process(proc):
    os.system(proc)


if __name__ == '__main__':
    main()
