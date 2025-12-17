import functools
import os
import copy
import json
import xml.etree.ElementTree as ET
from mmdet.utils import ConfigType
from mmdet.datasets import BaseDetDataset
from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
from mmyolo.registry import DATASETS
from mmengine.logging import MMLogger

@DATASETS.register_module()
class OWODDataset(BatchShapePolicyDataset, BaseDetDataset):
    """
    Standard Incremental Learning Dataset (No Unknown Detection).
    只关注已知类别，其余物体视为背景。
    """
    METAINFO = {
        'classes': (),
        'palette': None,
    }

    def __init__(self,
                 data_root: str,
                 dataset: str = 'MOWODB',
                 image_set: str = 'train',
                 task_id: int = 1,
                 task_metadata_path: str = '',
                 owod_cfg: ConfigType = None,
                 training_strategy: int = 0,
                 **kwargs):

        self.images = []
        self.annotations = []
        self.imgids = []
        self.imgid2annotations = {}
        self.image_set_fns = []

        self.image_set = image_set
        self.dataset = dataset
        self.task_id = task_id
        self.training_strategy = training_strategy
        self._logger = MMLogger.get_current_instance()

        use_dict_mode = len(task_metadata_path) > 0
        
        if use_dict_mode:
            with open(task_metadata_path, 'r') as f:
                self.task_metadata = json.load(f)
            
            task_key = f"t{task_id}"
            known_classes = self.task_metadata[task_key]["known"]
            cur_known_classes = self.task_metadata[task_key]["task_classes"]
            prev_classes = [cls for cls in known_classes if cls not in cur_known_classes]
            
            self.ordered_classes = prev_classes + cur_known_classes
            self.CLASS_NAMES = self.ordered_classes
            
            self.prev_intro_cls = len(prev_classes)
            self.cur_intro_cls = len(cur_known_classes)
            
            # 【修改点1】总类别数就是已知类别数，不再预留 Unknown 的位置
            self.total_num_class = len(self.CLASS_NAMES)
            self.num_seen_classes = len(self.ordered_classes)
            
            self._logger.info(f"📊 [Init] Task {task_id} ({image_set}):")
            self._logger.info(f"   Known Classes (Prev+Cur): {self.num_seen_classes} classes")
            
        else:
            # Legacy Mode
            assert owod_cfg is not None
            self.prev_intro_cls = owod_cfg.PREV_INTRODUCED_CLS
            self.cur_intro_cls = owod_cfg.CUR_INTRODUCED_CLS
            self.total_num_class = owod_cfg.num_classes
            self.num_seen_classes = self.prev_intro_cls + self.cur_intro_cls
            self._task_num = owod_cfg.task_num

        # Training strategy logging
        if "test" not in image_set:
            strategy_name = "ORACLE" if training_strategy == 1 else "INCREMENTAL"
            self._logger.info(f"🎯 Training strategy: {strategy_name}")

        OWODDataset.METAINFO['classes'] = self.CLASS_NAMES
        
        # ==================== 2. 图片文件加载 ====================
        self.data_root = str(data_root)
        annotation_dir = os.path.join(self.data_root, 'Annotations', dataset)
        image_dir = os.path.join(self.data_root, 'JPEGImages', dataset)

        file_names = self.extract_fns()
        
        self.image_set_fns.extend(file_names)
        self.images.extend([os.path.join(image_dir, x + ".jpg") for x in file_names])
        self.annotations.extend([os.path.join(annotation_dir, x + ".xml") for x in file_names])
        self.imgids.extend(x for x in file_names)            
        self.imgid2annotations.update(dict(zip(self.imgids, self.annotations)))

        if len(self.images) == 0:
            self._logger.warning(f"⚠️  No images found for Task {task_id} set {image_set}!")
        else:
            self._logger.info(f"✅ Loaded {len(self.images)} images for Task {task_id} (Set: {image_set})")

        super().__init__(**kwargs)

    def extract_fns(self):
        splits_dir = os.path.join(self.data_root, 'ImageSets', self.dataset)
        image_sets = []
        file_names = []

        target_split = f"t{self.task_id}_{self.image_set}"
        image_sets.append(target_split)
        
        self.image_set_list = image_sets
        
        for image_set in image_sets:
            split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
            if not os.path.exists(split_f):
                self._logger.warning(f"⚠️ Split file not found: {split_f}")
                continue
                
            with open(split_f, "r") as f:
                file_names.extend([x.strip() for x in f.readlines()])
        
        return file_names

    # ==================== Label 处理逻辑 (已简化) ====================
    
    def filter_instances(self, target, valid_class_range):
        """
        通用过滤函数：
        只保留 bbox_label 在 valid_class_range 范围内的实例。
        不在范围内的直接移除（视为背景）。
        """
        entry = copy.copy(target)
        # 必须倒序遍历或者使用新列表，否则 remove 会出错
        output_instances = []
        for annotation in entry:
            if annotation["bbox_label"] in valid_class_range:
                output_instances.append(annotation)
        return output_instances

    def load_data_list(self):
        data_list = []
        for i, img_id in enumerate(self.imgids):
            raw_data_info = dict(img_path=self.images[i], img_id=img_id)
            parsed_data_info = self.parse_data_info(raw_data_info)
            data_list.append(parsed_data_info)
        return data_list
    
    def parse_data_info(self, raw_data_info):
        data_info = copy.copy(raw_data_info)
        img_id = data_info["img_id"]
        tree = ET.parse(self.imgid2annotations[img_id])

        instances = []
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            
            # 【修改点2】解析阶段：如果 XML 里的类不在我们的 CLASS_NAMES 表里，直接跳过
            # 这意味着我们不关心任何“未知”类别，也不加载它们
            try:
                bbox_label = self.CLASS_NAMES.index(cls)
            except ValueError:
                continue 
            
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            
            instance = dict(
                bbox_label=bbox_label,
                bbox=bbox,
                area=(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                ignore_flag=0,
            )
            instances.append(instance)

        # ==================== 核心过滤逻辑 ====================
        # 根据不同的阶段，定义“有效类别范围”
        
        if 'train' in self.image_set:
            if self.training_strategy == 1:
                # Oracle 模式：看所有已知类
                valid_range = range(0, self.num_seen_classes)
            else:
                # 增量模式：只看当前任务引入的新类
                valid_range = range(self.prev_intro_cls, self.prev_intro_cls + self.cur_intro_cls)
            
            instances = self.filter_instances(instances, valid_range)
        
        elif 'test' in self.image_set:
            # 【修改点3】测试模式：只保留所有已见过的类 (0 ~ num_seen_classes)
            # 不再进行 Unknown 标记，超出范围的直接丢弃
            valid_range = range(0, self.num_seen_classes)
            instances = self.filter_instances(instances, valid_range)
            
        elif 'ft' in self.image_set:
            # Finetune 模式：看所有已知类
            valid_range = range(0, self.num_seen_classes)
            instances = self.filter_instances(instances, valid_range)
            
        data_info.update(
            height=int(tree.findall("./size/height")[0].text),
            width=int(tree.findall("./size/width")[0].text),
            instances=instances,
        )

        return data_info

    def filter_data(self):
        if self.test_mode:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
            if self.filter_cfg is not None else False
        min_size = self.filter_cfg.get('min_size', 0) \
            if self.filter_cfg is not None else 0

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and len(data_info['instances']) == 0:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos