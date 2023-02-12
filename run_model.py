import json
import logging
import os.path
import pickle
from typing import List

from PIL import ImageFile, Image
from transformers import HfArgumentParser

from vwsd.ltr import LTRModel
from vwsd.predictor import CLIPRetrievalPredictor
from vwsd.common import read_samples, ModelArgs, PredictionResult, translate_samples
from vwsd.wikipedia import Wikipedia, WikiImagePage

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 700_000_000


class ModelBuilder:

    def __init__(self, args: ModelArgs):
        assert args.multilingual_mode in ("original", "translated")
        if args.lang != "en":
            args.enable_t5 = False
        self.args = args
        self.args.original_lang = args.lang
        self.data_dir = os.path.join(args.data_dir, f"{args.data_split}_v1")
        self.images_dir = os.path.join(self.data_dir, f"{args.data_split}_images_v1")
        if args.data_split == "test":
            self.samples_path = os.path.join(self.data_dir, f"{args.lang}.test.data.v1.1.txt")
            gold_path = os.path.join(self.data_dir, f"{args.lang}.test.gold.v1.1.txt")
            if not os.path.exists(self.samples_path):
                self.samples_path = os.path.join(self.data_dir, f"{args.lang}.test.data.txt")
                gold_path = os.path.join(self.data_dir, f"{args.lang}.test.gold.txt")
            if os.path.exists(gold_path):
                self.gold_path = gold_path
            else:
                self.gold_path = None
            self.samples = read_samples(self.args, self.samples_path, self.gold_path, fix_test=True)
        else:
            self.samples_path = os.path.join(self.data_dir, f"{args.data_split}.data.v1.txt")
            self.gold_path = os.path.join(self.data_dir, f"{args.data_split}.gold.v1.txt")
            self.samples = read_samples(self.args, self.samples_path, self.gold_path)
        if args.lang != "en" and args.multilingual_mode == "translated":
            self.samples = translate_samples(self.samples, args)
            self.args.lang = "en"
        self.model: CLIPRetrievalPredictor = self._build_model(args)

    def _build_model(self, args: ModelArgs):
        wikipedia = Wikipedia(lang=args.lang, wit_path=args.wit_dir) if args.enable_wikipedia else None
        model = CLIPRetrievalPredictor(self.images_dir, wikipedia, args)
        if args.enable_weighting: model.enable_weighting(self.samples)
        return model

    def _train_ltr(self, samples: List[PredictionResult]):
        logging.info("Training LTR model")
        data_path = "ltr_dataset.pkl"
        with open(data_path, "wb") as output_file:
            pickle.dump(samples, output_file)
        ltr = LTRModel()
        ltr.train(data_path, self.args.ltr_path)

    def run(self):
        pred_path = f"prediction.{self.args.original_lang}.txt"
        log_path = f"runlog.{self.args.original_lang}.txt"
        with open(log_path, "w", encoding="utf-8") as runlog, open(pred_path, "w", encoding="utf-8") as pred:
            num_correct = 0
            mrr_sum = 0
            size = len(self.samples)
            correct_ranks = []
            outputs = []
            for idx, sample in enumerate(self.samples):
                output = self.model.predict(sample, self.images_dir)
                outputs.append(output)
                ranking = output.ranking
                pred.write("\t".join(ranking) + "\n")
                correct_idx = output.ranking.index(sample.gold) if sample.gold else -1
                correct_ranks.append(correct_idx + 1)
                mrr_sum += (1 / (correct_idx + 1)) if correct_idx >= 0 else 0
                if correct_idx == 0:
                    num_correct += 1
                acc = num_correct * 100 / (idx + 1)
                mrr = mrr_sum * 100 / (idx + 1)
                runlog.write(output.as_json() + "\n")
                message = f"pred: {output.label}, true: {sample.gold or '[unknown]'} [{idx+1}/{size}] "
                if self.gold_path is not None:
                    message += f"[acc: {acc:.2f}%, mrr: {mrr:.2f}%] "
                message += sample.context
                print(message)
        self.model.close()
        if self.gold_path is not None:
            print(f"Accuracy: {100 * num_correct / size:.4f}%, MRR: {100 * mrr_sum / size:.4f}%")
            top_k_acc = lambda k: 100 * sum([1 for val in correct_ranks if val <= k]) / size
            print(f"Top-K Accuracy, k=1: {top_k_acc(1):.2f}%, k=2: {top_k_acc(2):.2f}%, k=3: {top_k_acc(3):.2f}%")
        if self.args.train_ltr:
            self._train_ltr(outputs)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)
    logging.root.setLevel(logging.INFO)
    parser = HfArgumentParser(ModelArgs)
    model_args = parser.parse_args_into_dataclasses()[0]
    builder = ModelBuilder(model_args)
    builder.run()