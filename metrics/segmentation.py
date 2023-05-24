import numpy as np
from scipy import sparse
import torch

__all__ = ["IOU"]

class IOU(torch.nn.Module):
    def __init__(self, num_classes, exp_name=None, ignore_cls=None, weights=None, clazz_names=None, verbose=False):
        super().__init__()

        self.num_classes = num_classes
        self.exp_name = exp_name

        self.ignore_cls = ignore_cls
        self.verbose = verbose

        self.cm = np.zeros((num_classes, num_classes), 'u8')  # confusion matrix
        self.tps = np.zeros(num_classes, dtype='u8')  # true positives
        self.fps = np.zeros(num_classes, dtype='u8')  # false positives
        self.fns = np.zeros(num_classes, dtype='u8')  # false negatives
        self.weights = weights if weights is not None else np.ones(num_classes)  # Weights of each class for mean IOU
        self.clazz_names = clazz_names if clazz_names is not None else np.arange(num_classes)  # for nicer printing

    def forward(self, batch):
        infer = batch[self.infer_key]
        gt = batch[self.gt_key]

        self.update(labels=gt, predictions=infer)

    def update(self, labels, predictions):
        if type(labels) == torch.Tensor:
            labels = labels.detach().cpu().numpy()
            predictions = predictions.detach().cpu().numpy()

        if len(predictions.shape) > 1:
            predictions = np.argmax(predictions, 1)  # first dimension are probabilities/scores

        if self.ignore_cls is not None:
            mask = labels != self.ignore_cls
            labels = labels[mask]
            predictions = predictions[mask]

        tmp_cm = sparse.coo_matrix(
                (np.ones(np.prod(labels.shape), 'u8'), (labels.flatten(), predictions.flatten())),
                shape=(self.num_classes, self.num_classes)
        ).toarray()

        tps = np.diag(tmp_cm)
        fps = tmp_cm.sum(0) - tps
        fns = tmp_cm.sum(1) - tps
        self.cm += tmp_cm
        self.tps += tps
        self.fps += fps
        self.fns += fns

        if self.verbose:
            self.print_stats()

    def _compute_stats(self, tps, fps, fns):
        with np.errstate(
                all='ignore'):  # any division could be by zero, we don't really care about these errors, we know about these
            precisions = tps / (tps + fps)
            recalls = tps / (tps + fns)
            ious = tps / (tps + fps + fns)

        return precisions, recalls, ious

    def return_stats(self):
        with np.errstate(
                all='ignore'):  # any division could be by zero, we don't really care about these errors, we know about these
            precisions = self.tps / (self.tps + self.fps)
            recalls = self.tps / (self.tps + self.fns)
            ious = self.tps / (self.tps + self.fps + self.fns)

        return precisions, recalls, ious

    def get_precisions(self, interest_clz=[0]):
        precisions, recalls, ious = self._compute_stats(self.tps, self.fps, self.fns)
        res_li = []
        for c in interest_clz:
            res_li.append(np.array((precisions[c], recalls[c], ious[c], self.tps[c], self.fps[c], self.fns[c])))

        res_table = np.stack(res_li)
        return precisions

    def results_to_file(self, file=None):
        precisions, recalls, ious = self._compute_stats(self.tps, self.fps, self.fns)

        ious = np.insert(ious, len(ious), ious.mean())  # mean iou

        # if self.clazz_names[-1] != 'mIou':
        #     self.clazz_names.append('mIou')

        df = pd.DataFrame([precisions[1], recalls[1], ious[1]], index=["Precision", "Recall", "IOU"], columns=[0])
        print(df.to_latex())

        data = {'latex': df.to_latex(),
                'mIou': ious[-1],
                'clazz_names': self.clazz_names[:-1],
                'ious': ious[:-1],
                'precisions': precisions,
                'recalls': recalls,
                }

        if file is not None:
            np.savez(file, **data)

    def print_stats(self, classes=None):
        if classes is None:
            classes = range(self.num_classes)
        precisions, recalls, ious = self._compute_stats(self.tps, self.fps, self.fns)
        # print('\n---\n')
        for c in classes:
            print(
                    f'Class: {str(self.clazz_names[c]):20s}\t'
                    f'Precision: {precisions[c]:.3f}\t'
                    f'Recall {recalls[c]:.3f}\t'
                    f'IOU: {ious[c]:.3f}\t'

                    f'TP: {self.tps[c]:.3f}\t'
                    f'FP {self.fps[c]:.3f}\t'
                    f'FN: {self.fns[c]:.3f}\t'
            )
        # print(f"Mean IoU {ious.mean()}")
        # print('\n---\n')

    def reset(self):
        self.cm = np.zeros((self.num_classes, self.num_classes), 'u8')
        self.tps = np.zeros(self.num_classes, dtype='u8')
        self.fps = np.zeros(self.num_classes, dtype='u8')
        self.fns = np.zeros(self.num_classes, dtype='u8')

