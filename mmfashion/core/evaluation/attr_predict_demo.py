import numpy as np
import torch


class AttrPredictor(object):

    def __init__(self, cfg, tops_type=[3]):
        """Create the empty array to count true positive(tp),
            true negative(tn), false positive(fp) and false negative(fn).

        Args:
            class_num : number of classes in the dataset
            tops_type : default calculate top3, top5 and top10
        """

        attr_cloth_file = open(cfg.attr_cloth_file).readlines()
        self.attr_idx2name = {}
        self.attr_idx2type = {}
        for i, line in enumerate(attr_cloth_file[2:]):
            self.attr_idx2name[i], self.attr_idx2type[i] = line.strip('\n').split()
        
        self.typeid2name = {
            "1": "Print",
            "2": "Sleeve Length",
            "3": "Length",
            "4": "Neckline",
            "5": "Material",
            "6": "Fitting"
        }

        self.person_blacklist = []
        self.upper_blacklist = ["3"]
        self.lower_blacklist = ["2", "4"]

        self.tops_type = tops_type

    def print_attr_name(self, pred_idx):
        for idx in pred_idx:
            print(self.attr_idx2name[idx])

    def show_prediction(self, pred):
        if isinstance(pred, torch.Tensor):
            data = pred.data.cpu().numpy()
        elif isinstance(pred, np.ndarray):
            data = pred
        else:
            raise TypeError('type {} cannot be calculated.'.format(type(pred)))

        for i in range(pred.size(0)):
            indexes = np.argsort(data[i])[::-1]
            for topk in self.tops_type:
                idxes = indexes[:topk]
                print('[ Top%d Attribute Prediction ]' % topk)
                self.print_attr_name(idxes)
        
    def show_json(self, pred, class_name):
        if isinstance(pred, torch.Tensor):
            data = pred.data.cpu().numpy()
        elif isinstance(pred, np.ndarray):
            data = pred
        else:
            raise TypeError('type {} cannot be calculated.'.format(type(pred)))

        res = {}

        if class_name == "person":
            blacklist = self.person_blacklist
        elif class_name == "upper":
            blacklist = self.upper_blacklist
        elif class_name == "lower":
            blacklist = self.lower_blacklist

        for idx, typename in self.typeid2name.items():
            if not idx in blacklist:
                res[typename] = []

        for i in range(pred.size(0)):
            indexes = np.argsort(data[i])[::-1]
            for topk in self.tops_type:
                for idx in indexes:
                    confidence = float(data[i][idx])
                    if confidence > 0.5:
                        type_id = self.attr_idx2type[idx]
                        if not type_id in blacklist:
                            res[self.typeid2name[type_id]].append({
                                "label": self.attr_idx2name[idx],
                                "confidence": confidence
                            })
                    else:
                        break
        return res