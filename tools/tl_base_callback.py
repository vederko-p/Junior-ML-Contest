
from typing import Callable, List

from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

import model_selection.metrics as metrics
from tools.vectorizer import Vectorizer


class BaseCallBack:
    """Learning callback class.

        Parameters
        ----------
        writer : `from torch.utils.tensorboard.SummaryWriter`
            Writer instance.
        test_ds : `torch.utils.data.Dataset`
            Test dataset.
        loss_func : `Callable`
            Loss function.
        delimeter : `int`
            Save logs step.
        w_filepaths : `list`
            List of filepaths to save model weights.
        """

    def __init__(self,
                 writer: SummaryWriter,
                 test_ds: Dataset,
                 loss_func: Callable,
                 delimeter: int = 100,
                 w_filepaths: List[str] = None):
        self.test_ds = test_ds
        self.writer = writer
        self.loss_func = loss_func
        self.delimeter = delimeter
        self.w_filepaths = w_filepaths
        self.step = 0

    def __call__(self, model, loss):
        return self.forward(model, loss)

    def forward(self, model, loss):
        raise Exception('Not implemented method.')


class ClassifyCallBack(BaseCallBack):
    def __init__(self,
                 writer: SummaryWriter,
                 test_ds: Dataset,
                 loss_func: Callable,
                 delimeter: int = 100,
                 w_filepaths: List[str] = None):
        super().__init__(writer, test_ds, loss_func, delimeter, w_filepaths)
        self.accuracy = metrics.CustomAccuracy()
        self.f1_score = metrics.F1Score()

    def forward(self, model, loss):
        self.step += 1
        self.writer.add_scalar('LOSS/train', loss, self.step)
        if not self.step % self.delimeter:
            # Ошибка:
            y_true, y_pred = metrics.eval_classif_ds(self.test_ds, model,
                                                     probs=True)
            test_loss = self.loss_func(y_pred, y_true)
            self.writer.add_scalar('LOSS/test', test_loss, self.step)
            # Метрики:
            acc = self.accuracy.ds_scope(self.test_ds, model)
            f1_score = self.f1_score.ds_scope(self.test_ds, model)
            self.writer.add_scalar('REPORT/accuracy', acc, self.step)
            self.writer.add_scalar('REPORT/f1_score', f1_score, self.step)
            # Веса:
            if self.w_filepaths is not None:
                model.split_save(self.w_filepaths)


class TripletLossCallBack(BaseCallBack):
    def __init__(self,
                 writer: SummaryWriter,
                 test_ds: Dataset,
                 loss_func: Callable,
                 delimeter: int = 100,
                 w_filepaths: List[str] = None):
        super().__init__(writer, test_ds, loss_func, delimeter, w_filepaths)
        self.vectorizer = Vectorizer()
        self.var_crit = metrics.CustomVarianceCriteria()

        self.best_var_crit = 1e7

    def forward(self, model, loss):
        self.step += 1
        self.writer.add_scalar('LOSS/train', loss, self.step)
        if not self.step % self.delimeter:
            # Метрики:
            self.vectorizer.vectorize(self.test_ds, model)
            var_crit = self.var_crit.scope(self.vectorizer.vects_res,
                                           self.vectorizer.lables_res)
            self.writer.add_scalar('REPORT/var_crit', var_crit, self.step)
            # Веса:
            if var_crit < self.best_var_crit:
                self.best_var_crit = var_crit

                best_var_fn = [f'{fn[:-3]}_best_var.pt' for
                               fn in self.w_filepaths]
                model.split_save(best_var_fn)
            if self.w_filepaths is not None:
                last_fn = [f'{fn[:-3]}_last.pt' for
                           fn in self.w_filepaths]
                model.split_save(last_fn)
