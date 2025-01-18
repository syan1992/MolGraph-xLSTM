import numpy as np
from numpy import array
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict

import torch

class Evaluator:
    """Evaluator for the prediction performance."""

    def __init__(self, name: str = "hiv", num_tasks: int = 1, eval_metric: str = "rocauc"):
        """
        Args:
            name (str, optional): The name of the dataset. Defaults to "hiv".
            num_tasks (int, optional): Number of tasks in the dataset. Defaults to 1.
            eval_metric (str, optional): Metrics for the evaluation. Defaults to "rocauc".
                Metrics include : 'rocauc', 'ap', 'rmse', 'mae', 'acc', 'F1'.
        """
        self.name = name
        self.num_tasks = num_tasks
        self.eval_metric = eval_metric

    def _parse_and_check_input(self, input_dict: Dict[np.ndarray, np.ndarray]):
        """Evaluate the performance of the input_dict.

        Args:
            input_dict (Dict[np.ndarray, np.ndarray]): The true value and the predict
                value of the dataset. The format of input_dict is like:
                input_dict = {"y_true": y_true, "y_pred": y_pred}.

        Returns:
            y_true, y_pred: The true value and the predict value of the dataset.
        """
        if (
            self.eval_metric == "rocauc"
            or self.eval_metric == "ap"
            or self.eval_metric == "rmse"
            or self.eval_metric == "mae"
            or self.eval_metric == "acc"
        ):
            if not "y_true" in input_dict:
                raise RuntimeError("Missing key of y_true")
            if not "y_pred" in input_dict:
                raise RuntimeError("Missing key of y_pred")

            y_true, y_pred = input_dict["y_true"], input_dict["y_pred"]

            """
                y_true: numpy ndarray or torch tensor of shape (num_graph, num_tasks)
                y_pred: numpy ndarray or torch tensor of shape (num_graph, num_tasks)
            """

            # converting to torch.Tensor to numpy on cpu
            if isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()

            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()

            ## check type
            if not isinstance(y_true, np.ndarray):
                raise RuntimeError(
                    "Arguments to Evaluator need to be either numpy ndarray or torch tensor"
                )

            if not y_true.shape == y_pred.shape:
                raise RuntimeError("Shape of y_true and y_pred must be the same")

            if not y_true.ndim == 2:
                raise RuntimeError(
                    "y_true and y_pred mush to 2-dim arrray, {}-dim array given".format(
                        y_true.ndim
                    )
                )

            if not y_true.shape[1] == self.num_tasks:
                raise RuntimeError(
                    "Number of tasks for {} should be {} but {} given".format(
                        self.name, self.num_tasks, y_true.shape[1]
                    )
                )

            return y_true, y_pred

        elif self.eval_metric == "F1":
            if not "seq_ref" in input_dict:
                raise RuntimeError("Missing key of seq_ref")
            if not "seq_pred" in input_dict:
                raise RuntimeError("Missing key of seq_pred")

            seq_ref, seq_pred = input_dict["seq_ref"], input_dict["seq_pred"]

            if not isinstance(seq_ref, list):
                raise RuntimeError("seq_ref must be of type list")

            if not isinstance(seq_pred, list):
                raise RuntimeError("seq_pred must be of type list")

            if len(seq_ref) != len(seq_pred):
                raise RuntimeError("Length of seq_true and seq_pred should be the same")

            return seq_ref, seq_pred

        else:
            raise ValueError("Undefined eval metric %s " % (self.eval_metric))

    def eval(self, input_dict: Dict[np.ndarray, np.ndarray]):
        """Evaluate the performance of the input_dict.

        Args:
            input_dict (Dict[np.ndarray, np.ndarray]): The true value and the predict
                value of the dataset. The format of input_dict is like:
                input_dict = {"y_true": y_true, "y_pred": y_pred}

        Returns:
            A scalar value of the selected metric.
        """
        if self.eval_metric == "rocauc":
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_rocauc(y_true, y_pred)
        if self.eval_metric == "ap":
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_ap(y_true, y_pred)
        elif self.eval_metric == "rmse":
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_rmse(y_true, y_pred)
        elif self.eval_metric == "mae":
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_mae(y_true, y_pred)
        elif self.eval_metric == "acc":
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_acc(y_true, y_pred)
        elif self.eval_metric == "F1":
            seq_ref, seq_pred = self._parse_and_check_input(input_dict)
            return self._eval_F1(seq_ref, seq_pred)
        else:
            raise ValueError("Undefined eval metric %s " % (self.eval_metric))

    @property
    def expected_input_format(self):
        desc = "==== Expected input format of Evaluator for {}\n".format(self.name)
        if self.eval_metric == "rocauc" or self.eval_metric == "ap":
            desc += "{'y_true': y_true, 'y_pred': y_pred}\n"
            desc += "- y_true: numpy ndarray or torch tensor of shape (num_graph, num_task)\n"
            desc += "- y_pred: numpy ndarray or torch tensor of shape (num_graph, num_task)\n"
            desc += "where y_pred stores score values (for computing AUC score),\n"
            desc += "num_task is {}, and ".format(self.num_tasks)
            desc += "each row corresponds to one graph.\n"
            desc += "nan values in y_true are ignored during evaluation.\n"
        elif self.eval_metric == "rmse":
            desc += "{'y_true': y_true, 'y_pred': y_pred}\n"
            desc += "- y_true: numpy ndarray or torch tensor of shape (num_graph, num_task)\n"
            desc += "- y_pred: numpy ndarray or torch tensor of shape (num_graph, num_task)\n"
            desc += "where num_task is {}, and ".format(self.num_tasks)
            desc += "each row corresponds to one graph.\n"
            desc += "nan values in y_true are ignored during evaluation.\n"
        elif self.eval_metric == "acc":
            desc += "{'y_true': y_true, 'y_pred': y_pred}\n"
            desc += "- y_true: numpy ndarray or torch tensor of shape (num_node, num_task)\n"
            desc += "- y_pred: numpy ndarray or torch tensor of shape (num_node, num_task)\n"
            desc += "where y_pred stores predicted class label (integer),\n"
            desc += "num_task is {}, and ".format(self.num_tasks)
            desc += "each row corresponds to one graph.\n"
        elif self.eval_metric == "F1":
            desc += "{'seq_ref': seq_ref, 'seq_pred': seq_pred}\n"
            desc += "- seq_ref: a list of lists of strings\n"
            desc += "- seq_pred: a list of lists of strings\n"
            desc += "where seq_ref stores the reference sequences of sub-tokens, and\n"
            desc += "seq_pred stores the predicted sequences of sub-tokens.\n"
        else:
            raise ValueError("Undefined eval metric %s " % (self.eval_metric))

        return desc

    @property
    def expected_output_format(self):
        desc = "==== Expected output format of Evaluator for {}\n".format(self.name)
        if self.eval_metric == "rocauc":
            desc += "{'rocauc': rocauc}\n"
            desc += "- rocauc (float): ROC-AUC score averaged across {} task(s)\n".format(
                self.num_tasks
            )
        elif self.eval_metric == "ap":
            desc += "{'ap': ap}\n"
            desc += (
                "- ap (float): Average Precision (AP) score averaged across {} task(s)\n".format(
                    self.num_tasks
                )
            )
        elif self.eval_metric == "rmse":
            desc += "{'rmse': rmse}\n"
            desc += "- rmse (float): root mean squared error averaged across {} task(s)\n".format(
                self.num_tasks
            )
        elif self.eval_metric == "acc":
            desc += "{'acc': acc}\n"
            desc += "- acc (float): Accuracy score averaged across {} task(s)\n".format(
                self.num_tasks
            )
        elif self.eval_metric == "F1":
            desc += "{'F1': F1}\n"
            desc += "- F1 (float): F1 score averaged over samples.\n"
        else:
            raise ValueError("Undefined eval metric %s " % (self.eval_metric))

        return desc

    def _eval_rocauc(self, y_true: np.ndarray, y_pred: np.ndarray):
        """compute ROC-AUC averaged across tasks.

        Args:
            y_true (np.ndarray): The true label of the dataset.
            y_pred (np.ndarray): The predict label of the dataset.
        """
        rocauc_list = []

        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                # ignore nan values
                is_labeled = y_true[:, i] == y_true[:, i]
                rocauc_list.append(roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i]))

        if len(rocauc_list) == 0:
            raise RuntimeError("No positively labeled data available. Cannot compute ROC-AUC.")

        return {"rocauc": sum(rocauc_list) / len(rocauc_list)}

    def _eval_ap(self, y_true: np.ndarray, y_pred: np.ndarray):
        """compute Average Precision (AP) averaged across tasks.

        Args:
            y_true (np.ndarray): The true label of the dataset.
            y_pred (np.ndarray): The predict label of the dataset.
        """

        ap_list = []

        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                # ignore nan values
                is_labeled = y_true[:, i] == y_true[:, i]
                ap = average_precision_score(y_true[is_labeled, i], y_pred[is_labeled, i])

                ap_list.append(ap)

        if len(ap_list) == 0:
            raise RuntimeError(
                "No positively labeled data available. Cannot compute Average Precision."
            )

        return {"ap": sum(ap_list) / len(ap_list)}

    def _eval_rmse(self, y_true: np.ndarray, y_pred: np.ndarray):
        """compute RMSE averaged across tasks.

        Args:
            y_true (np.ndarray): The true label of the dataset.
            y_pred (np.ndarray): The predict label of the dataset.
        """
        rmse_list = []

        for i in range(y_true.shape[1]):
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            rmse_list.append(np.sqrt(((y_true[is_labeled] - y_pred[is_labeled]) ** 2).mean()))

        return {"rmse": sum(rmse_list) / len(rmse_list)}

    def _eval_mae(self, y_true: np.ndarray, y_pred: np.ndarray):
        """compute MAE averaged across tasks.

        Args:
            y_true (np.ndarray): The true label of the dataset.
            y_pred (np.ndarray): The predict label of the dataset.
        """
        from sklearn.metrics import mean_absolute_error

        mae_list = []
        for i in range(y_true.shape[1]):
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            mae_list.append(mean_absolute_error(y_true[is_labeled], y_pred[is_labeled]))
        return {"mae": sum(mae_list) / len(mae_list)}

    def _eval_acc(self, y_true: array, y_pred: array):
        """compute accuracy averaged across tasks.

        Args:
            y_true (np.ndarray): The true label of the dataset.
            y_pred (np.ndarray): The predict label of the dataset.
        """
        acc_list = []

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:, i] == y_true[:, i]
            correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
            acc_list.append(float(np.sum(correct)) / len(correct))

        return {"acc": sum(acc_list) / len(acc_list)}

    def _eval_F1(self, seq_ref: np.ndarray, seq_pred: np.ndarray):
        """compute F1 score averaged across tasks.

        Args:
            seq_ref (np.ndarray): The true label of the dataset.
            seq_pred (np.ndarray): The predict label of the dataset.
        """

        precision_list = []
        recall_list = []
        f1_list = []

        for l, p in zip(seq_ref, seq_pred):
            label = set(l)
            prediction = set(p)
            true_positive = len(label.intersection(prediction))
            false_positive = len(prediction - label)
            false_negative = len(label - prediction)

            if true_positive + false_positive > 0:
                precision = true_positive / (true_positive + false_positive)
            else:
                precision = 0

            if true_positive + false_negative > 0:
                recall = true_positive / (true_positive + false_negative)
            else:
                recall = 0
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        return {
            "precision": np.average(precision_list),
            "recall": np.average(recall_list),
            "F1": np.average(f1_list),
        }
