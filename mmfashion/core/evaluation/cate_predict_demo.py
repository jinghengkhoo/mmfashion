import numpy as np
import torch


class CatePredictor(object):

    def __init__(self, cfg, tops_type=[3]):
        """Create the empty array to count true positive(tp),
            true negative(tn), false positive(fp) and false negative(fn).

        Args:
            class_num : number of classes in the dataset
            tops_type : default calculate top3, top5 and top10
        """

        cate_cloth_file = open(cfg.cate_cloth_file).readlines()
        self.cate_idx2name = {}
        self.cate_idx2type = {}
        for i, line in enumerate(cate_cloth_file[2:]):
            self.cate_idx2name[i], self.cate_idx2type[i] = line.strip('\n').split()

        self.tops_type = tops_type

    def print_cate_name(self, pred_idx):
        for idx in pred_idx:
            print(self.cate_idx2name[idx])

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
                print('[ Top%d Category Prediction ]' % topk)
                self.print_cate_name(idxes)

    def show_json(self, pred, class_name):
        if isinstance(pred, torch.Tensor):
            data = pred.data.cpu().numpy()
        elif isinstance(pred, np.ndarray):
            data = pred
        else:
            raise TypeError('type {} cannot be calculated.'.format(type(pred)))

        res = {"Category": []}

        if class_name == "person":
            valid_id = "3"
        elif class_name == "upper":
            valid_id = "1"
        elif class_name == "lower":
            valid_id = "2"

        for i in range(pred.size(0)):
            indexes = np.argsort(data[i])[::-1]
            for topk in self.tops_type:
                for idx in indexes:
                    confidence = float(data[i][idx])
                    if confidence > 0.5:
                        type_id = self.cate_idx2type[idx]
                        if type_id == valid_id:
                            res["Category"].append({
                                "label": self.cate_idx2name[idx],
                                "confidence": confidence
                            })
                    else:
                        break
                    
        for name, lis in res.items():
            if not lis:
                del res[name]

        return res