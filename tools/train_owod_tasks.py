import os.path as osp
import glob
from pathlib import Path
import subprocess
import re
import shutil
from mmengine.config import Config

# OWODB settings
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
    "ODinW13" : {
        "task_list" : [0,5,12,13,14,16,17,35,36,39,44,45,46,46],
        "test_image_set": "all_task_test",
    }    
}

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Train OWOD tasks')
    parser.add_argument('dataset', type=str, choices=["MOWODB", "SOWODB", "nuOWODB","ODinW13"])
    parser.add_argument('config', type=str)
    parser.add_argument('ckpt', type=str)
    parser.add_argument('--start', type=int, default=1, help='Start task number')
    parser.add_argument('--threshold', type=float, default=0.05, help='Confidence score threshold for known class')
    parser.add_argument('--suffix', type=str, help='Suffix for work_dir')
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

def get_latest_best_ckpt(work_dir):
    ckpts = glob.glob(osp.join(work_dir, "best*.pth"))
    if not ckpts:
        return None

    def extract_epoch_num(filename):
        match = re.search(r'epoch[_\-]?(\d+)', osp.basename(filename))
        return int(match.group(1)) if match else -1

    ckpts.sort(key=extract_epoch_num)
    latest_best = ckpts[-1]
    print(f"🟩 Found best checkpoint: {osp.basename(latest_best)} (epoch={extract_epoch_num(latest_best)})")
    return latest_best
import shlex
def run_dataset(dataset, config, load_from, args,abs_running_path):
    stem = Path(config).stem
    task_num = len(owod_settings[dataset]['task_list'])
    prev_work_dir = ''
    cfg = Config.fromfile(args.config)

    for task in range(args.start, task_num):
        work_dir = f'{cfg.WORK_DIR}/{stem}_{dataset.lower()}_train_task{task}'
        if args.suffix:
            work_dir += f"_{args.suffix}"

        command = (f'DATASET={dataset} TASK={task} THRESHOLD={args.threshold} SAVE={args.save} CUDA_VISIBLE_DEVICES=0,1,2 '
                   f'./tools/dist_train_owod.sh {config} 3 --amp --work-dir {work_dir} --cfg-options model.all_class_embeddings_path={shlex.quote(abs_running_path)}')

        if task > 1:
            if not prev_work_dir:
                prev_work_dir = f'{cfg.WORK_DIR}/{stem}_{dataset.lower()}_train_task{task-1}'
                
            best_ckpt = get_latest_best_ckpt(prev_work_dir)
            if best_ckpt:
                load_from = best_ckpt
            else:
                pass
                #load_from = sorted(glob.glob(osp.join(prev_work_dir, "best*.pth")))[-1]

        if load_from:
            command += f" --cfg-options load_from={load_from} "

        if args.save:
            with open('eval_outputs.txt', 'a') as f:
                f.write(f"{dataset} [Task {task}] (thresh: {args.threshold}) - {stem}\n")

        print("<TRAIN>:", command)
        return_code = run_command(command)
        if return_code != 0:
            print(f"Task {task} failed with return code {return_code}")
            break

        # Update prev_work_dir
        prev_work_dir = work_dir

        
if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)

    abs_ckpt_path = str(Path(cfg.CKPT_PATH).expanduser().resolve())
    abs_ckpt_running = str(Path(cfg.CKPT_RUNNING).expanduser().resolve())
    abs_ckpt_final = str(Path(cfg.CKPT_FINAL).expanduser().resolve())

    Path(abs_ckpt_running).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(abs_ckpt_path, abs_ckpt_running)

    # run_dataset should accept abs_running_path and pass it into command as cfg-option
    run_dataset(args.dataset, args.config, args.ckpt, args, abs_ckpt_running)

    shutil.copy2(abs_ckpt_running, abs_ckpt_final)
        
