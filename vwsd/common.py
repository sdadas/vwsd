import json
import math
import os
import logging
import pickle
from abc import ABC, abstractmethod
from collections import Counter

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import List, Dict, Optional
import numpy as np
import torch.cuda
from PIL import Image
from dataclasses import dataclass, field
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration


@dataclass
class ModelArgs:
    data_dir: str = field(
        metadata={"help": "Path to the VWSD data directory"},
    )
    data_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use"},
    )
    wit_dir: str = field(
        default=None,
        metadata={"help": "Path to the WIT dataset directory"},
    )
    clip_model: str =  field(
        default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        metadata={"help": "The name or path to the model"},
    )
    clip_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size for the CLIP model"},
    )
    enable_wikipedia: bool = field(
        default=True,
        metadata={"help": "Enable Wikipedia retrieval module"},
    )
    enable_weighting: bool = field(
        default=True,
        metadata={"help": "Enable image weighting module"},
    )
    enable_wordnet: bool = field(
        default=True,
        metadata={"help": "Enable Wordnet context expansion module"},
    )
    enable_t5: bool = field(
        default=False,
        metadata={"help": "Enable T5 context expansion module"},
    )
    t5_model: str = field(
        default="google/flan-t5-xxl",
        metadata={"help": "Name of the T5 model to use for expansion"},
    )
    train_ltr: bool = field(
        default=False,
        metadata={"help": "Train learning to rank model"},
    )
    use_ltr: bool = field(
        default=True,
        metadata={"help": "Use learning to rank model for prediction"},
    )
    ltr_path: str = field(
        default="ltr_model.json",
        metadata={"help": "Path to the trained LTR model"},
    )
    lang: str = field(
        default="en",
        metadata={"help": "Task language"},
    )
    original_lang: str = field(
        default=None,
        metadata={"help": "Task original language"},
    )
    multilingual_mode: str = field(
        default="original",
        metadata={"help": "Approach for multilingual inference, possible values are: original, translated"},
    )

    @property
    def ltr(self):
        return self.train_ltr or self.use_ltr

    @property
    def translation_model(self):
        if self.lang == "fa":
            return "persiannlp/mt5-large-parsinlu-opus-translation_fa_en"
        elif self.lang == "it":
            return "Helsinki-NLP/opus-mt-tc-big-it-en"
        else:
            return None

    @property
    def text_encoder_model(self):
        if self.lang == "fa":
            return "semeval-vwsd-clip-fa"
        elif self.lang == "it":
            return "semeval-vwsd-clip-it"
        else:
            return None

    def safe_model_name(self):
        return self.clip_model.replace("/", "_").replace(".", "_")


class NumpyJSONEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class InputSample:

    def __init__(self, word: str, context: str, images: List[str], gold: str=None):
        self.word = word
        self.context = context
        self.images = images
        self.gold = gold
        self.context_word = self.compute_context_word()
        self.t5_context = None
        self.sense_id = None
        self.explanation = {}

    def compute_context_word(self):
        word = self.word.lower()
        context_words = list(self.context.lower().split())
        try:
            context_words.remove(word)
        except ValueError:
            logging.warning("Fixing incorrect context '%s' missing target word '%s'", self.context, self.word)
            context_words = list(self.context.lower().split())
            context_words.append(word)
            self.context = self.context + " " + word
            context_words.remove(word)
        return " ".join(context_words)

    def read_images(self, images_dir: str):
        return [Image.open(os.path.join(images_dir, image)) for image in self.images]

    def explain(self, component: str, **kwargs):
        exp = self.explanation.get(component, {})
        exp.update(kwargs)
        self.explanation[component] = exp
        return self


class PredictionResult:

    def __init__(self , sample: InputSample):
        self.sample = sample
        self.scores = np.zeros(len(sample.images))
        self.model_name = None

    def set_scores(self, scores: np.ndarray, model_name: str=None):
        self.scores = scores.flatten()
        self.model_name = model_name
        return self

    @property
    def label(self) -> str:
        predicted_idx = np.argmax(self.scores)
        return self.sample.images[predicted_idx]

    @property
    def ranking(self) -> List[str]:
        ranking = sorted(zip(self.sample.images, self.scores), key=lambda v: -v[1])
        return [val[0] for val in ranking]

    def as_json(self):
        res = dict(self.__dict__)
        res["label"] = self.label
        res["sample"] = dict(self.sample.__dict__)
        return json.dumps(res, cls=NumpyJSONEncoder)


class CLIP:

    def __init__(self, args: ModelArgs):
        self.args = args
        self.model_name = args.clip_model
        model, processor = self._load_model(args.clip_model)
        self.model: CLIPModel = model
        self.processor: CLIPProcessor = processor
        self.text_encoder: Optional[SentenceTransformer] = self._load_text_encoder(args)

    def _load_model(self, clip_model: str):
        model = CLIPModel.from_pretrained(clip_model)
        processor = CLIPProcessor.from_pretrained(clip_model)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        return model, processor

    def _load_text_encoder(self, args: ModelArgs):
        encoder_name = args.text_encoder_model
        if encoder_name is None: return None
        model = SentenceTransformer(encoder_name)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        return model

    def embed_texts(self, texts: List[str], tqdm_desc: str = None, normalize: bool = True):
        text_vectors = []
        for i in tqdm(range(0, len(texts), self.args.clip_batch_size), disable=tqdm_desc is None, desc=tqdm_desc):
            batch = texts[i:i + self.args.clip_batch_size]
            embeds = self._embed_texts_clip(batch) if self.text_encoder is None else self._embed_texts_external(batch)
            if normalize:
                embeds = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)
            text_vectors.append(embeds)
        return np.vstack(text_vectors)

    def _embed_texts_clip(self, batch: List[str]) -> np.ndarray:
        inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77)
        if torch.cuda.is_available(): inputs = inputs.to("cuda")
        outputs = self.model.get_text_features(**inputs)
        return outputs.detach().cpu().numpy()

    def _embed_texts_external(self, batch: List[str]):
        return self.text_encoder.encode(batch, convert_to_numpy=True, show_progress_bar=False)

    def embed_images(self, images: List, batch_size: int, normalize: bool = True) -> np.ndarray:
        if len(images) <= batch_size:
            return self._embed_images_batch(images, normalize)
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            results.append(self._embed_images_batch(batch, normalize))
        return np.vstack(results)

    def _embed_images_batch(self, images, normalize: bool = True) -> np.ndarray:
        if len(images) == 0: return np.array([])
        inputs = self.processor(images=images, return_tensors="pt", padding=True, truncation=True)
        if torch.cuda.is_available(): inputs = inputs.to("cuda")
        outputs = self.model.get_image_features(**inputs)
        image_embeds = outputs.detach().cpu().numpy()
        if normalize:
            image_embeds = image_embeds / np.linalg.norm(image_embeds, axis=1, keepdims=True)
        return image_embeds


class ScoreWeighting(ABC):

    @abstractmethod
    def apply(self, sample: InputSample, scores: np.ndarray):
        raise NotImplementedError()


class DatasetStats:

    def __init__(self):
        self.total_count = 0
        self.image_frequency = {}
        self.context_frequency = {}
        self.all_senses = []
        self.all_contexts = []
        self.all_images = []
        self.sense_image_sim = {}
        self.context_image_sim = {}
        self.penalties = {}


class DatasetsStatsWeighting(ScoreWeighting):

    def __init__(self, dataset: List, model: CLIP, image_cache: Dict, args: ModelArgs):
        self.args = args
        self.stats: DatasetStats = self._compute_stats(dataset, model, image_cache)

    def apply(self, sample: InputSample, scores: np.ndarray):
        penalties = np.array([self.stats.penalties[image] for image in sample.images])
        sense_sim = np.array([self.stats.sense_image_sim[(sample.word, image)] for image in sample.images])
        ctx_sim = np.array([self.stats.context_image_sim[(sample.context_word, image)] for image in sample.images])
        img_frq = np.array([math.log10(self.stats.image_frequency[image]) for image in sample.images])
        ctx_frq = math.log10(self.stats.context_frequency[sample.context_word])
        sample.explain("stats", values=penalties, sense_sim=sense_sim, ctx_sim=ctx_sim, img_frq=img_frq, ctx_frq=ctx_frq)
        results = scores - penalties
        return results

    def _compute_stats(self, dataset: List, model: CLIP, img_cache: Dict):
        cached = self._load_from_cache() if self.args.original_lang == "en" else None
        if cached is not None: return cached
        result = DatasetStats()
        self._compute_counts(result, dataset)
        all_images = []
        all_image_vectors = []
        for key, val in img_cache.items():
            all_images.append(key)
            all_image_vectors.append(val)
        all_image_vectors = np.vstack(all_image_vectors)
        result.all_images = all_images
        self._compute_text2img_sim(dataset, result, model, all_image_vectors)
        self._compute_penalties(dataset, result, model, all_image_vectors)
        if self.args.original_lang == "en":
            self._save_to_cache(result)
        return result

    def _compute_counts(self, result: DatasetStats, dataset: List):
        image_frequency = Counter()
        context_frequency = Counter()
        all_senses = set()
        all_contexts = set()
        for sample in dataset:
            all_senses.add(sample.word)
            all_contexts.add(sample.context_word)
            context_frequency[sample.context_word] += 1
            for image in sample.images:
                image_frequency[image] += 1
        result.all_senses = sorted(list(all_senses))
        result.all_contexts = sorted(list(all_contexts))
        result.image_frequency = image_frequency
        result.context_frequency = context_frequency
        result.total_count = len(dataset)

    def _compute_penalties(self, dataset: List, result: DatasetStats, model: CLIP, all_image_vectors):
        texts = [sample.context for sample in dataset]
        all_text_vectors = model.embed_texts(texts, tqdm_desc="Computing text embeddings")
        all_pairs_similarity = np.matmul(all_text_vectors, all_image_vectors.transpose())
        max_frq = result.image_frequency.most_common(n=1)[0][1]
        image_scores = all_pairs_similarity.mean(axis=0).tolist()
        image_scores_data = [
            (key, val, result.image_frequency[key] / max_frq)
            for key, val in zip(result.all_images, image_scores)
        ]
        image_scores_data = sorted(image_scores_data, key=lambda v: -(v[1] * v[2]))
        penalties = {key: score * frq for key, score, frq in image_scores_data}
        result.penalties = penalties

    def _compute_text2img_sim(self, dataset: List, result: DatasetStats, model: CLIP, all_image_vectors):
        all_sense_vectors = model.embed_texts(result.all_senses, tqdm_desc="Computing sense embeddings")
        all_context_vectors = model.embed_texts(result.all_contexts, tqdm_desc="Computing context embeddings")
        senses = {val: idx for idx, val in enumerate(result.all_senses)}
        contexts = {val: idx for idx, val in enumerate(result.all_contexts)}
        images = {val: idx for idx, val in enumerate(result.all_images)}
        sense_image_sim = np.matmul(all_sense_vectors, all_image_vectors.transpose())
        context_image_sim = np.matmul(all_context_vectors, all_image_vectors.transpose())
        for sample in dataset:
            sense_idx = senses[sample.word]
            context_word = sample.context_word
            contex_idx = contexts[context_word]
            for image in sample.images:
                image_idx = images[image]
                result.sense_image_sim[(sample.word, image)] = sense_image_sim[sense_idx][image_idx]
                result.context_image_sim[(context_word, image)] = context_image_sim[contex_idx][image_idx]

    def _load_from_cache(self):
        data_dir = os.path.join(self.args.data_dir, f"{self.args.data_split}_v1")
        cached_path = os.path.join(data_dir, f"{self.args.lang}_stats_{self.args.safe_model_name()}.bin")
        if os.path.exists(cached_path):
            logging.info("Found cached image weights, loading from cache")
            with open(cached_path, "rb") as input_file:
                return pickle.load(input_file)
        else: return None

    def _save_to_cache(self, result: DatasetStats):
        data_dir = os.path.join(self.args.data_dir, f"{self.args.data_split}_v1")
        cached_path = os.path.join(data_dir, f"{self.args.lang}_stats_{self.args.safe_model_name()}.bin")
        with open(cached_path, "wb") as output_file:
            pickle.dump(result, output_file)


class NoOpWeighting(ScoreWeighting):

    def apply(self, sample: InputSample, scores: np.ndarray):
        return scores


class Predictor(ABC):

    @abstractmethod
    def predict(self, sample: InputSample, images_dir: str) -> PredictionResult:
        raise NotImplementedError()


class T5ContextExpansion:

    def __init__(self, args: ModelArgs, cached_path: str):
        self.args = args
        self.cached_path = cached_path
        self.template = "What is the meaning of \"{}\"?"

    def expand_contexts(self, samples: List[InputSample], batch_size: int = 4):
        if os.path.exists(self.cached_path):
            logging.info("Found expanded context in cache, loading file %s", self.cached_path)
            with open(self.cached_path, "r", encoding="utf-8") as input_file:
                return [line.strip() for line in input_file]
        logging.info("Loading %s model for context expansion", self.args.t5_model)
        model, tokenizer = self._load_t5()
        results = []
        for i in tqdm(range(0, len(samples), batch_size)):
            batch = samples[i:i + batch_size]
            results.extend(self._expand_batch(batch, model, tokenizer))
        with open(self.cached_path, "w", encoding="utf-8") as output_file:
            output_file.writelines([result + "\n" for result in results])
        del model
        del tokenizer
        torch.cuda.empty_cache()
        return results

    def _load_t5(self):
        model = T5ForConditionalGeneration.from_pretrained(self.args.t5_model)
        tokenizer = AutoTokenizer.from_pretrained(self.args.t5_model)
        if torch.cuda.is_available():
            model.bfloat16()
            model.eval()
            model.cuda()
        else:
            model.eval()
        return model, tokenizer

    def _expand_batch(self, batch: List[InputSample], model: T5ForConditionalGeneration, tokenizer):
        src_texts = [self.template.format(val.context) for val in batch]
        input_tokens = tokenizer(src_texts, return_tensors="pt", padding=True, truncation=True)
        if torch.cuda.is_available():
            input_tokens = input_tokens.to("cuda")
        output_tokens = model.generate(
            **input_tokens,
            max_new_tokens=30,
            num_return_sequences=1,
            num_beams=5
        )
        return [tokenizer.decode(t, skip_special_tokens=True) for t in output_tokens]


def read_samples(args: ModelArgs, samples_path: str, gold_path: str = None, fix_test: bool = False):
    results: List[InputSample] = []
    with open(samples_path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            if fix_test:
                line = line.replace("adobe", "abode").replace("ecstacy", "ecstasy")
            row = [val.strip().replace("\u2002", "-") for val in line.strip().split("\t")]
            sample = InputSample(row[0], row[1], row[2:])
            results.append(sample)
    if gold_path is not None:
        with open(gold_path, "r", encoding="utf-8") as input_file:
            for idx, line in enumerate(input_file):
                row = results[idx]
                gold = line.strip()
                row.gold = gold
    if args.enable_t5:
        cached_path = samples_path.replace(".txt", "_t5_contexts.txt")
        t5 = T5ContextExpansion(args, cached_path)
        contexts = t5.expand_contexts(results)
        assert len(contexts) == len(results)
        for idx, sample in enumerate(results):
            sample.t5_context = contexts[idx]
    return results


def translate_samples(samples: List[InputSample], args: ModelArgs, batch_size: int = 8):
    logging.info("Translating dataset from '%s' to 'en'", args.original_lang)
    model_name = args.translation_model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if torch.cuda.is_available(): model.cuda()
    model.eval()
    texts = [f"{val.context}, {val.word}" for val in samples]
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        input_tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        if torch.cuda.is_available(): input_tokens = input_tokens.to("cuda")
        output_tokens = model.generate(**input_tokens)
        results.extend([tokenizer.decode(t, skip_special_tokens=True) for t in output_tokens])
    for idx, sample in enumerate(samples):
        result = results[idx]
        if result.count(",") != 1:
            logging.error("Incorrect translation '%s' -> '%s'", texts[idx], result)
            continue
        parts = [val.strip() for val in result.split(",")]
        sample.context = parts[0]
        sample.word = parts[1].replace(" ", "-")
        sample.compute_context_word()
    del model, tokenizer
    return samples
