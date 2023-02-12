import logging
import os
import pickle
from typing import List, Dict, Optional
import numpy as np
from nltk.corpus import wordnet as wn
from nltk import download as ntlk_download
from tqdm import tqdm
from PIL import Image
import torch.cuda

from vwsd.common import InputSample, PredictionResult, Predictor, ModelArgs, CLIP, NoOpWeighting, DatasetsStatsWeighting
from vwsd.ltr import LTRModel
from vwsd.wikipedia import Wikipedia
from vwsd.wordnet import WordnetSense


class CLIPCachedPredictor(Predictor):

    def __init__(self, images_dir: str, args: ModelArgs):
        self.args = args
        self.clip = CLIP(args)
        self.images_dir = images_dir
        self.image_cache: Dict = self._load_images(images_dir)
        self.weighting = NoOpWeighting()

    def _load_images(self, images_dir: str):
        model_name = self.args.safe_model_name()
        cache_path = os.path.join(images_dir, os.pardir, f"vectors_{model_name}.bin")
        if os.path.exists(cache_path):
            logging.info("Found cached vectors, loading from cache")
            with open(cache_path, "rb") as cache_file:
                return pickle.load(cache_file)
        files = os.listdir(images_dir)
        batch = []
        res = {}
        for filename in tqdm(files, desc="Loading image embeddings"):
            try:
                image = Image.open(os.path.join(images_dir, filename))
            except Exception as ex:
                print(filename)
                raise ex
            batch.append((filename, image))
            if len(batch) >= self.args.clip_batch_size:
                self._load_image_batch(batch, res)
                batch = []
        if len(batch) > 0: self._load_image_batch(batch, res)
        with open(cache_path, "wb") as cache_file:
            logging.info("Saving vectors to cache")
            pickle.dump(res, cache_file)
        return res

    def _load_image_batch(self, batch: List, res: Dict):
        images = [image for _, image in batch]
        inputs = self.clip.processor(images=images, return_tensors="pt", padding=True)
        if torch.cuda.is_available(): inputs = inputs.to("cuda")
        outputs = self.clip.model.get_image_features(**inputs)
        image_embeds = outputs.detach().cpu().numpy()
        for idx, item in enumerate(batch):
            filename = item[0]
            embed = image_embeds[idx, :]
            res[filename] = embed / np.linalg.norm(embed, keepdims=True)

    def predict(self, sample: InputSample, images_dir: str) -> PredictionResult:
        texts = [sample.context]
        text_embeds = self.clip.embed_texts(texts)
        image_embeds = self.get_sample_embeddings(sample)
        scores = self.weighting.apply(sample, np.matmul(text_embeds, image_embeds.transpose()))
        predicted_idx = np.argmax(scores, axis=1)[0]
        label = sample.images[predicted_idx]
        sample.explain("clip", model=self.args.clip_model, label=label, probs=scores.flatten())
        return PredictionResult(sample).set_scores(scores, "clip")

    def get_sample_embeddings(self, sample: InputSample):
        return np.vstack([self.image_cache[fname] for fname in sample.images])


class CLIPRetrievalPredictor(Predictor):

    def __init__(self, images_dir: str, wikipedia: Optional[Wikipedia], args: ModelArgs):
        self.args = args
        self.clip = CLIPCachedPredictor(images_dir, args)
        self.wikipedia = wikipedia
        self.ltr = LTRModel(args.ltr_path) if args.use_ltr else LTRModel()
        self.wn_langs = {
            "en": ["eng", "eng_wikt", "eng_cldr"],
            "it": ["ita", "ita_iwn", "ita_wikt", "ita_cldr"],
            "fa": ["fas_wikt", "fas_cldr"]
        }[self.args.lang]
        ntlk_download('wordnet')
        ntlk_download('omw-1.4')
        ntlk_download("extended_omw")
        wn.add_exomw()

    def enable_weighting(self, dataset: List[InputSample]):
        self.clip.weighting = DatasetsStatsWeighting(dataset, self.clip.clip, self.clip.image_cache, self.args)

    def predict(self, sample: InputSample, images_dir: str) -> PredictionResult:
        wiki_prediction = self._retrieve_from_wikipedia(sample)
        if wiki_prediction is not None and not self.args.ltr: return wiki_prediction
        context = self._augment_context(sample) if self.args.enable_wordnet else sample.context
        sample.explain("wordnet", original_context=sample.context, augmented=context != sample.context, context=context)
        sample.original_context = sample.context
        sample.context = context
        clip_prediction = self.clip.predict(sample, images_dir)
        result = wiki_prediction if wiki_prediction else clip_prediction
        if self.args.use_ltr:
            return self.ltr.predict(result)
        else:
            return result

    def _retrieve_from_wikipedia(self, sample: InputSample) -> Optional[PredictionResult]:
        if self.wikipedia is None:
            sample.explain("wiki", best_score=0.0, max_other_score=0.0, probs=np.zeros((10,)))
            return None
        sample_embeddings = self.clip.get_sample_embeddings(sample)
        scores = self.wikipedia.score(sample, sample_embeddings, self.clip.clip)
        predicted_idx = np.argmax(scores)
        best_score = scores[predicted_idx]
        scores[predicted_idx] = 0.0
        max_other_score = np.max(scores)
        scores[predicted_idx] = best_score
        label = sample.images[predicted_idx]
        sample.explain("wiki", label=label, best_score=best_score, max_other_score=max_other_score, probs=scores)
        if best_score > 0.9 and max_other_score < 0.8:
            return PredictionResult(sample).set_scores(scores, "wiki")
        else:
            return None

    def _augment_context(self, sample: InputSample):
        context = self._augment_context_by_wn(sample)
        if context is not None and context != sample.context:
            return context
        if self.args.enable_t5 and sample.t5_context is not None:
            return ", ".join([sample.context, sample.t5_context])
        return sample.context

    def _augment_context_by_wn(self, sample: InputSample):
        word = sample.word.lower()
        context_words = set(sample.context.lower().split())
        context_words.remove(word)
        current_context = sample.context
        lang = self.args.lang
        senses: List[WordnetSense] = []
        for wn_lang in self.wn_langs:
            senses.extend([WordnetSense(wn, wn_lang, syn, lang) for syn in wn.synsets(word, lang=wn_lang)])
        max_score = 0.0
        for sense in senses:
            score = sense.score_sense(context_words)
            if score > max_score:
                max_score = score
                current_context = sense.extract_context(sample)
                sample.sense_id = sense.id
        if lang != "en" and not max_score > 0.0:
            soft_scores = []
            soft_senses = []
            for sense in senses:
                score = sense.score_sense_vectors(context_words)
                if score > 0.0:
                    soft_scores.append(score)
                    soft_senses.append(sense)
            if len(soft_scores) > 0:
                best_idx = np.argmax(soft_scores)
                best_sense = soft_senses[best_idx]
                current_context = best_sense.extract_context(sample)
                sample.sense_id = best_sense.id
        if max_score > 0.0:
            logging.info("Augmented context: %s", current_context)
        return current_context

    def close(self):
        if self.wikipedia is not None:
            self.wikipedia.close()
