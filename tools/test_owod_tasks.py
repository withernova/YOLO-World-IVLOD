import os.path as osp
import glob
from pathlib import Path
import subprocess
from mmengine.config import Config

import shlex
owod_settings = {
    # 4 tasks
    "MOWODB": {
        "task_list": [0, 20, 40, 60, 80],
        "test_image_set": "all_task_test"
    },
    # 4 tasks
    "SOWODB": {
        "task_list": [0, 19, 40, 60, 80],
        "test_image_set": "test",
    },
    # 3 tasks
    "nuOWODB": {
        "task_list": [0, 10, 17, 23],
        "test_image_set": "test",
    },
    # "ODinW13" : {
    #     "task_list" : [0,5,12,13,14,16,17,35,36,39,42,43,44,44],
    #     "test_image_set": "test",
    # },
    "ODinW13" : {
        "task_list" : [0,5,7,1,1,2,1,20,1,3,5,1,1,2],
        "test_image_set": "test",
    },       
    "ZCOCO" :{
        "task_list" : [0,80],
        "test_image_set" : "test_all"
    }
}
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate OWOD tasks')
    parser.add_argument('dataset', type=str, choices=["MOWODB", "SOWODB", "nuOWODB","ZCOCO","ODinW13"])
    parser.add_argument('config', type=str)
    parser.add_argument('ckpt', type=str)
    parser.add_argument('--threshold', type=float, default=0.05, help='Confidence score threshold for known class')
    parser.add_argument('--save', action='store_true', help='Save evaluation results to eval_output.txt')
    return parser.parse_args()


def run_command(command):
    process = subprocess.Popen(["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in process.stdout:
        print(line, end="")
    for line in process.stderr:
        print(line, end="")
    return_code = process.wait()
    return return_code


def eval_dataset(dataset, config, ckpt, args,abs_running_path):
    image_set = owod_settings[dataset]['test_image_set']
    task_num = len(owod_settings[dataset]['task_list'])
    cfg = Config.fromfile(config)

    for task in range(1, task_num):
        stem = Path(config).stem
        work_dir = f'{cfg.WORK_DIR}/{stem}_{dataset.lower()}_test_task{task}'
        # ckpt_path = osp.join(ckpt.format(stem, dataset.lower(), task), "best*.pth")
        # checkpoint = sorted(glob.glob(ckpt_path))[-1]

        command = (f"DATASET={dataset} TASK={task} THRESHOLD={args.threshold} SAVE={args.save} "
                   f"./tools/dist_test.sh {config} {ckpt} 1 --work-dir {work_dir} --json-prefix {work_dir}/test_results/result --cfg-options model.all_class_embeddings_path={shlex.quote(abs_running_path)}")

        if args.save:
            with open('eval_outputs.txt', 'a') as f:
                f.write(f"Eval {dataset} [Task {task}]\n")

        print("<EVAL>:", command)
        return_code = run_command(command)
        if return_code != 0:
            print(f"Task {task} failed with return code {return_code}")
            break

import shutil
if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    abs_ckpt_path = str(Path(cfg.CKPT_PATH).expanduser().resolve())
    abs_ckpt_running = str(Path(cfg.CKPT_RUNNING).expanduser().resolve())
    Path(abs_ckpt_running).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(abs_ckpt_path, abs_ckpt_running)
    
    eval_dataset(args.dataset, args.config, args.ckpt, args,abs_ckpt_running)