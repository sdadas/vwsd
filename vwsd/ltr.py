import logging
import pickle
import random
from typing import List, Tuple
import xgboost as xgb
import numpy as np

from vwsd.common import PredictionResult


class LTRModel:

    def __init__(self, load_from_path: str=None):
        self.model = None
        if load_from_path is not None:
            logging.info(f"Loading XGBRanker from path {load_from_path}")
            self.model = xgb.XGBRanker()
            self.model.load_model(load_from_path)

    def train(self, data_path: str, model_path: str):
        with open(data_path, "rb") as input_file:
            data = pickle.load(input_file)
            self.fit(data, model_path)

    def predict(self, pred: PredictionResult):
        x, y, _ = self._build_data_points(pred)
        output = self.model.predict(x)
        pred.set_scores(np.array(output), "ltr")
        return pred

    def fit(self, data: List[PredictionResult], model_path: str, valid_fraction: float=0.1):
        grouped = [self._build_data_points(val) for val in data]
        random.shuffle(grouped)
        valid_num = int(len(data) * valid_fraction)
        train = grouped[:-valid_num] if valid_fraction > 0 else grouped
        valid = grouped[-valid_num:] if valid_fraction > 0 else None
        X_train = np.vstack([x for x, y, g in train])
        y_train = np.hstack([y for x, y, g in train])
        groups = [g for x, y, g in train]
        model = self._train_model(X_train, y_train, groups)
        if valid is not None:
            self._validate_model(model, valid)
        model.save_model(model_path)
        logging.info(f"Saved LTR model to {model_path}")

    def _train_model(self, X_train, y_train, groups):
        model = xgb.XGBRanker(
            tree_method="gpu_hist",
            booster="gbtree",
            objective="rank:pairwise",
            random_state=42,
            learning_rate=0.1,
            colsample_bytree=0.9,
            eta=0.05,
            max_depth=6,
            n_estimators=110,
            subsample=0.75
        )
        model.fit(X_train, y_train, group=groups, verbose=True)
        return model

    def _validate_model(self, model: xgb.XGBRanker, grouped: List[Tuple]):
        num_correct = 0
        for idx, group in enumerate(grouped):
            x, y, _ = group
            pred = model.predict(x)
            true_idx = int(np.argmax(y))
            pred_idx = int(np.argmax(pred))
            if pred_idx == true_idx:
                num_correct += 1
        print(f"Accuracy: {100 * num_correct / len(grouped):.4f}%")

    def _build_data_points(self, pred: PredictionResult) -> Tuple:
        num_images = range(len(pred.sample.images))
        points = [self._build_data_point(idx, pred) for idx in num_images]
        if pred.sample.gold is not None:
            ranking = pred.ranking
            gold = pred.sample.gold
            ranking.remove(gold)
            relevance = [(1 if val == gold else 0) for val in pred.sample.images]
        else:
            relevance = [0 for _ in pred.sample.images]
        return np.vstack(points), np.array(relevance), len(pred.sample.images)

    def _build_data_point(self, idx: int, pred: PredictionResult):
        result = []
        exp = pred.sample.explanation
        penalties = exp["stats"]
        result.append(penalties["values"][idx])
        result.append(penalties["sense_sim"][idx])
        result.append(penalties["ctx_sim"][idx])
        result.append(penalties["img_frq"][idx])
        result.append(penalties["ctx_frq"])
        self._build_score_stats(idx, result, exp["wiki"]["probs"])
        self._build_score_stats(idx, result, exp["clip"]["probs"])
        return np.array(result)

    def _build_score_stats(self, idx: int, result: List, scores: np.ndarray):
        score = scores[idx]
        scores[idx] = 0.0
        max_other_score = np.max(scores)
        mean_other_scores = np.mean(scores)
        diff_max = score - max_other_score
        diff_mean = score - mean_other_scores
        scores[idx] = score
        result.extend([score, max_other_score, mean_other_scores, diff_max, diff_mean])


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)
    logging.root.setLevel(logging.INFO)
    model = LTRModel()
    model.train("ltr_dataset.pkl", "ltr_model.json")