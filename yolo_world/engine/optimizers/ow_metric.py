import os
import sys
import tempfile
import copy
import json
import logging
from typing import Optional
import numpy as np
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from functools import lru_cache

from mmyolo.registry import METRICS
from mmdet.utils import ConfigType
from mmengine.logging import MMLogger
from mmengine.evaluator import BaseMetric

np.set_printoptions(threshold=sys.maxsize, linewidth=200)

def bbox_iou_numpy(box1, box2):
    """计算两个 numpy bbox 数组之间的 IoU"""
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    lt = np.maximum(box1[:, None, :2], box2[:, :2])  # [N, M, 2]
    rb = np.minimum(box1[:, None, 2:], box2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clip(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter
    return inter / np.maximum(union, 1e-6)

@METRICS.register_module()
class OpenWorldMetric(BaseMetric):
    """
    Standard Pascal VOC style Evaluator with Confusion Matrix.
    """
    default_prefix: Optional[str] = 'standard_eval'

    def __init__(self,
                 data_root: str,
                 dataset_name: str,
                 task_id: int = 1,
                 split: str = 'test',
                 task_metadata_path: str = '',
                 threshold: float = 0.0,
                 save_rets: bool = False,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 outfile_prefix: Optional[str] = None, 
                 **kwargs):  
        
        super().__init__(collect_device=collect_device, prefix=prefix)

        self._logger = MMLogger.get_current_instance()
        self.threshold = threshold
        self.save_rets = save_rets
        self.outfile_prefix = outfile_prefix
        self._is_2007 = False

        # 加载任务元数据
        if len(task_metadata_path) > 0:
            with open(task_metadata_path, 'r') as f:
                self.task_metadata = json.load(f)
            
            task_key = f"t{task_id}"
            known_classes = self.task_metadata[task_key]["known"]
            cur_known_classes = self.task_metadata[task_key]["task_classes"]
            prev_classes = [cls for cls in known_classes if cls not in cur_known_classes]
            
            self.ordered_classes = prev_classes + cur_known_classes
            self.known_classes = self.ordered_classes
            self._class_names = self.ordered_classes 
            
            self.prev_intro_cls = len(prev_classes)
            self.cur_intro_cls = len(cur_known_classes)
            self.num_seen_classes = len(self.ordered_classes)
            
            self._logger.info(f"📊 [Standard Mode] Metric for Task {task_id}:")
            self._logger.info(f"   Target Classes: {self.num_seen_classes}")
            self._logger.info(f"   Class List: {self.known_classes}")

        self._anno_file_template = os.path.join(data_root, "Annotations", dataset_name, "{}.xml")
        self._image_set_path = os.path.join(data_root, "ImageSets", dataset_name, f"t{task_id}_{split}.txt")
        
        self.confusion_data = [] 

    def process(self, data_batch: dict, data_samples):
        """Process one batch of data samples and predictions."""
        for data_sample in data_samples:
            pred = data_sample['pred_instances']
            pred_bboxes = pred['bboxes'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()

            det = []
            valid_mask = (pred_labels >= 0) & (pred_labels < self.num_seen_classes)
            
            cm_pred_bboxes = []
            cm_pred_labels = []
            cm_pred_scores = []

            for i, (box, score, label) in enumerate(zip(pred_bboxes, pred_scores, pred_labels)):
                if not valid_mask[i]: continue
                
                if score > max(0.05, self.threshold):
                    cm_pred_bboxes.append(box)
                    cm_pred_labels.append(label)
                    cm_pred_scores.append(score)

                if score > self.threshold:
                    xmin, ymin, xmax, ymax = box
                    det.append([label, data_sample['img_id'], score, xmin+1, ymin+1, xmax+1, ymax+1])
            
            self.results.append(det)

            gt = data_sample['gt_instances']
            gt_bboxes = gt['bboxes'].cpu().numpy()
            gt_labels = gt['labels'].cpu().numpy()
            
            gt_valid_mask = (gt_labels >= 0) & (gt_labels < self.num_seen_classes)
            gt_bboxes = gt_bboxes[gt_valid_mask]
            gt_labels = gt_labels[gt_valid_mask]

            self.confusion_data.append({
                'gt_bboxes': gt_bboxes,
                'gt_labels': gt_labels,
                'pred_bboxes': np.array(cm_pred_bboxes) if len(cm_pred_bboxes) > 0 else np.zeros((0, 4)),
                'pred_labels': np.array(cm_pred_labels) if len(cm_pred_labels) > 0 else np.zeros((0,)),
                'pred_scores': np.array(cm_pred_scores) if len(cm_pred_scores) > 0 else np.zeros((0,))
            })

    def compute_metrics(self, results: list):
        """Compute standard AP/AR and Confusion Matrix."""
        
        predictions = defaultdict(list)
        for dets in results:
            for det in dets:
                cls, image_id, score, xmin, ymin, xmax, ymax = det
                predictions[cls].append(
                    f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                )

        with tempfile.TemporaryDirectory(prefix="voc_eval_standard_") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")
            aps = defaultdict(list)
            recs = defaultdict(list)
            iou_thresholds = range(50, 100, 5)
            
            per_class_ap50 = {}

            for cls_id, cls_name in enumerate(self._class_names):
                lines = predictions.get(cls_id, [])
                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))

                for thresh in iou_thresholds:
                    rec, prec, ap = voc_eval(
                        res_file_template, self._anno_file_template, self._image_set_path,
                        cls_name, ovthresh=thresh / 100.0, use_07_metric=self._is_2007,
                        known_classes=self.known_classes
                    )
                    aps[thresh].append(ap * 100)
                    
                    if thresh == 50:
                        per_class_ap50[cls_name] = ap * 100

                    try:
                        recs[thresh].append(rec[-1] * 100)
                    except:
                        recs[thresh].append(0.0)

        def get_avg(metric_dict, iou_key, indices):
            if not indices: return 0.0
            values = [metric_dict[iou_key][i] for i in indices]
            return np.mean(values)

        def get_coco_map(metric_dict, indices):
            if not indices: return 0.0
            all_aps = []
            for t in iou_thresholds:
                all_aps.extend([metric_dict[t][i] for i in indices])
            return np.mean(all_aps)

        all_indices = range(self.prev_intro_cls, self.num_seen_classes)
        mAP_50 = get_avg(aps, 50, all_indices)
        mAP_75 = get_avg(aps, 75, all_indices)
        mAP_coco = get_coco_map(aps, all_indices)
        mAR_50 = get_avg(recs, 50, all_indices)
        mAR_75 = get_avg(recs, 75, all_indices)
        mAR_coco = get_coco_map(recs, all_indices)

        self._logger.info("\n" + "="*20 + " Standard Detection Results " + "="*20)
        self._logger.info(f"Classes: {self.num_seen_classes} known classes")
        
        # === 新增：打印逐类别的 AP50 ===
        self._logger.info("\n--- Per Class AP (IoU=0.50) ---")
        header = f"{'Class Name':<20} | {'AP50':<10}"
        self._logger.info(header)
        self._logger.info("-" * len(header))
        for name in self._class_names:
            val = per_class_ap50.get(name, 0.0)
            self._logger.info(f"{name:<20} | {val:.2f}")
        self._logger.info("-" * len(header))

        self._logger.info("\n--- Precision (AP) ---")
        self._logger.info(f"mAP (IoU=0.50): {mAP_50:.2f}")
        self._logger.info(f"mAP (IoU=0.75): {mAP_75:.2f}")
        self._logger.info(f"mAP (0.5:0.95): {mAP_coco:.2f}")
        self._logger.info("\n--- Recall (AR) ---")
        self._logger.info(f"mAR (IoU=0.50): {mAR_50:.2f}")
        self._logger.info(f"mAR (IoU=0.75): {mAR_75:.2f}")
        self._logger.info(f"mAR (0.5:0.95): {mAR_coco:.2f}")

        # === 计算并打印混淆矩阵 ===
        self._calculate_and_print_confusion_matrix()

        ret = {
            "mAP": mAP_coco, "mAP_50": mAP_50, "mAP_75": mAP_75,
            "mAR_50": mAR_50, "mAR_coco": mAR_coco,
        }
        
        # 如果你想把逐类别的 AP 也放到返回值里（可选）
        # for k, v in per_class_ap50.items():
        #     ret[f"AP50_{k}"] = v
            
        return ret

    def _calculate_and_print_confusion_matrix(self):
        """
        计算混淆矩阵。
        Row: Ground Truth
        Col: Prediction
        """
        self._logger.info("\n--- Confusion Matrix Analysis (IoU=0.5) ---")
        
        num_classes = self.num_seen_classes
        # Matrix size: (Classes + Background) x (Classes + Background)
        matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
        
        iou_thresh = 0.5
        conf_thresh = 0.3 

        for data in self.confusion_data:
            gt_bboxes = data['gt_bboxes']
            gt_labels = data['gt_labels']
            
            pred_bboxes = data['pred_bboxes']
            pred_labels = data['pred_labels']
            pred_scores = data['pred_scores']

            # 过滤低分预测
            keep = pred_scores > conf_thresh
            pred_bboxes = pred_bboxes[keep]
            pred_labels = pred_labels[keep]

            if len(gt_bboxes) == 0 and len(pred_bboxes) == 0:
                continue
            
            if len(gt_bboxes) == 0:
                for pl in pred_labels:
                    matrix[num_classes, pl] += 1
                continue

            if len(pred_bboxes) == 0:
                for gl in gt_labels:
                    matrix[gl, num_classes] += 1
                continue

            # 计算 IoU 矩阵 [N_gt, M_pred]
            ious = bbox_iou_numpy(gt_bboxes, pred_bboxes)
            
            # 贪婪匹配
            gt_matched = np.zeros(len(gt_bboxes), dtype=bool)
            pred_matched = np.zeros(len(pred_bboxes), dtype=bool)
            
            if ious.size > 0:
                flatten_ious = ious.flatten()
                indices = np.argsort(-flatten_ious)
                
                for idx in indices:
                    iou = flatten_ious[idx]
                    if iou < iou_thresh:
                        break
                        
                    gt_idx = idx // len(pred_bboxes)
                    pred_idx = idx % len(pred_bboxes)
                    
                    if not gt_matched[gt_idx] and not pred_matched[pred_idx]:
                        gt_l = gt_labels[gt_idx]
                        pred_l = pred_labels[pred_idx]
                        matrix[gt_l, pred_l] += 1
                        
                        gt_matched[gt_idx] = True
                        pred_matched[pred_idx] = True

            # 统计未匹配
            for i, matched in enumerate(gt_matched):
                if not matched:
                    matrix[gt_labels[i], num_classes] += 1
            
            for i, matched in enumerate(pred_matched):
                if not matched:
                    matrix[num_classes, pred_labels[i]] += 1

        # === 打印逻辑 ===
        names = self._class_names + ['Background']
        
        # 1. Top 混淆对
        self._logger.info("Top Confusion Pairs (GT -> Pred):")
        confusion_list = []
        for i in range(num_classes):
            for j in range(num_classes):
                if i != j and matrix[i, j] > 0:
                    confusion_list.append((matrix[i, j], names[i], names[j]))
        
        confusion_list.sort(key=lambda x: x[0], reverse=True)
        for count, gt_n, pred_n in confusion_list[:10]: 
            self._logger.info(f"  {gt_n} -> {pred_n}: {count} times")

        # 2. 背景误检
        self._logger.info("\nTop Background False Positives (Background -> Pred):")
        bg_fp = []
        for j in range(num_classes):
            count = matrix[num_classes, j]
            if count > 0:
                bg_fp.append((count, names[j]))
        bg_fp.sort(key=lambda x: x[0], reverse=True)
        for count, pred_n in bg_fp[:10]:
            self._logger.info(f"  Background -> {pred_n}: {count} times")

        # 3. 漏检
        self._logger.info("\nTop Missed Classes (GT -> Background):")
        missed = []
        for i in range(num_classes):
            count = matrix[i, num_classes]
            if count > 0:
                missed.append((count, names[i]))
        missed.sort(key=lambda x: x[0], reverse=True)
        for count, gt_n in missed[:10]:
            self._logger.info(f"  {gt_n} -> Missed: {count} times")


@lru_cache(maxsize=None)
def parse_rec(filename, known_classes):
    """
    Parse a PASCAL VOC xml file.
    """
    tree = ET.parse(filename)
    objects = []
    known_set = set(known_classes)

    for obj in tree.findall("object"):
        cls_name = obj.find("name").text
        if cls_name not in known_set:
            continue
        
        obj_struct = {}
        obj_struct["name"] = cls_name
        obj_struct['difficult'] = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text), int(bbox.find("ymin").text),
            int(bbox.find("xmax").text), int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)
    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall."""
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False, known_classes=None):
    """Standard VOC evaluation for a single class."""
    with open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    imagenames_filtered = []
    recs = {}
    mapping = {}
    for imagename in imagenames:
        rec = parse_rec(annopath.format(imagename), tuple(known_classes))
        if rec is not None and int(imagename) not in mapping:
            recs[imagename] = rec
            imagenames_filtered.append(imagename)
            mapping[int(imagename)] = imagename

    imagenames = imagenames_filtered
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool_)
        det = [False] * len(R)
        npos += sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}
    
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    if not lines:
        return [], [], 0.0

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)

    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        if int(image_ids[d]) not in mapping:
            fp[d] = 1.0
            continue
            
        R = class_recs[mapping[int(image_ids[d])]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            uni = ((bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0) - inters)
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0 
        else:
            fp[d] = 1.0 

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / (float(npos) + 1e-5)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap