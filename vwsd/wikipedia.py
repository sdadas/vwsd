import csv
import hashlib
import json
import logging
import os
import shelve
import string
import sys
from pathlib import Path
from typing import Iterable, List, Dict, Optional

import mwparserfromhell
import numpy as np
import requests
from gensim.scripts.segment_wiki import segment_all_articles
from tqdm import tqdm
from unidecode import unidecode
from PIL import Image

from vwsd.common import InputSample, CLIP


class WikiImagePage:

    def __init__(self, title: str):
        self.title = title
        self.images = []
        self.embeddings: Optional[Dict] = None


class WikiImageCache:

    def __init__(self, lang: str, wit_path: str):
        self.lang = lang
        self.wit_path = wit_path
        self.wit: shelve.Shelf = self._load_wit_dataset(wit_path)
        self.filename_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
        self.filename_chars = frozenset(self.filename_chars)

    def _load_wit_dataset(self, wit_path: str) -> shelve.Shelf:
        logging.info("Loading WIT dataset from path %s", wit_path)
        cached_path = os.path.join(wit_path, f"cached_{self.lang}.db")
        cache = shelve.open(cached_path)
        if len(cache) > 0: return cache
        files = os.listdir(wit_path)
        for file in files:
            if file.endswith(".tsv"):
                self._load_wit_file(cache, os.path.join(wit_path, file))
        return cache

    def _load_wit_file(self, results: shelve.Shelf, wit_path: str):
        with open(wit_path, "r", encoding="utf-8", newline="") as wit_file:
            reader = csv.DictReader(wit_file, delimiter="\t")
            for row in tqdm(reader, desc=wit_path):
                lang = row["language"]
                if lang != self.lang: continue
                page = row["page_title"]
                result = results.get(page, WikiImagePage(page))
                result.images.append(row)
                results[page] = result

    def find_images(self, page_title: str, model: CLIP) -> Optional[WikiImagePage]:
        res: WikiImagePage = self.wit.get(page_title, None)
        if res is None: return res

        embedding_dict = isinstance(res.embeddings, dict)
        if res.embeddings is None or not embedding_dict:
            self._cache_images(res, model)
        elif embedding_dict and model.model_name not in res.embeddings:
            self._cache_images(res, model)
        return res

    def _cache_images(self, page: WikiImagePage, model: CLIP):
        if not isinstance(page.embeddings, dict):
            page.embeddings = dict()
        model_name = model.model_name
        logging.info("Computing image embeddings for page %s", page.title)
        page_title = self._safe_name(page.title)
        images_dir = os.path.join(self.wit_path, f"image_cache_{self.lang}", page_title)
        os.makedirs(images_dir, exist_ok=True)
        images = []
        for image in page.images:
            image_path = self._download_image(image, images_dir)
            image["error"] = image_path is None
            if image_path is not None:
                try:
                    image = Image.open(image_path)
                    images.append(image)
                except Exception as ex:
                    logging.error("Error opening image %s", repr(ex))
        try:
            images_embeds = model.embed_images(images, batch_size=8)
            page.embeddings[model_name] = images_embeds
        except OSError as ex:
            logging.error("Error computing embeddings %s", repr(ex))
            page.embeddings[model_name] = np.array([])
        self.wit[page.title] = page

    def _download_image(self, image: Dict, images_dir: str):
        image_url = image["image_url"]
        image_name = self._safe_name(image_url.split("/")[-1])
        extension = Path(image_name).suffix
        if len(image_name) > 200:
            image_name = hashlib.md5(image_name.encode("utf-8")).hexdigest() + extension
        if extension in (".svg",):
            return None
        image_path = os.path.join(images_dir, image_name)
        if os.path.exists(image_path):
            return image_path
        with requests.get(image_url, stream=True, headers={"User-Agent": "WIT Downloader"}) as r:
            try:
                r.raise_for_status()
            except Exception as ex:
                logging.error("Error downloading image %s", repr(ex))
                return None
            with open(image_path, 'wb') as output_file:
                for chunk in r.iter_content(chunk_size=8192):
                    output_file.write(chunk)
        return image_path

    def _safe_name(self, value: str):
        return "".join(c for c in unidecode(value) if c in self.filename_chars)

    def close(self):
        self.wit.close()


class WikiIndex:

    def __init__(self, lang: str, service_url: str = "http://localhost:8080"):
        self.lang = lang
        self.service_url = service_url.strip("/")

    def search(self, query: str)  -> Dict:
        url = f"{self.service_url}/{self.lang}/search"
        headers = {"content-type": "application/json;charset=UTF-8"}
        body = json.dumps({"query": query}, ensure_ascii=False)
        response = requests.post(url, data=body.encode("utf-8"), headers=headers)
        return response.json()

    def build_index(self, dump_path: str, batch_size: int=1000):
        logging.info("Building index from path %s", dump_path)
        total_articles = self._get_total_articles_in_wikipedia()
        batch = []
        for article in tqdm(self._get_articles(dump_path), total=total_articles, unit="pages", smoothing=0.1):
            batch.append(article)
            if len(batch) > batch_size:
                self._send_batch(batch)
                batch = []
        self._send_batch(batch)

    def _get_total_articles_in_wikipedia(self):
        url = f"https://{self.lang}.wikipedia.org/w/api.php?action=query&meta=siteinfo&siprop=statistics&format=json"
        response = requests.get(url)
        value = json.loads(response.content)
        return value["query"]["statistics"]["pages"]

    def _send_batch(self, batch: List):
        if len(batch) == 0: return
        url = f"{self.service_url}/{self.lang}/add"
        body = json.dumps(batch, ensure_ascii=False)
        headers = {"content-type": "application/json;charset=UTF-8"}
        response = requests.post(url, data=body.encode("utf-8"), headers=headers)
        assert response.status_code == 200

    def _get_articles(self, wiki_path: str):
        article_stream = segment_all_articles(wiki_path, 15, include_interlinks=True)
        for idx, article in enumerate(article_stream):
            article_title, article_sections = article[0], article[1]
            yield self._parse_article(article_title, article_sections)

    def _parse_article(self, title: str, sections: Iterable):
        text = []
        first_section = None
        for section_heading, section_code in sections:
            parsed = mwparserfromhell.parse(section_code)
            section_text = parsed.strip_code().strip()
            if len(section_text) == 0: continue
            if first_section is None: first_section = section_text
            text.append(section_text)
        return {"title": title, "summary": first_section, "text": " ".join(text)}


class Wikipedia:

    def __init__(self, lang: str, wit_path: str, service_url: str="http://localhost:8080"):
        self.index = WikiIndex(lang, service_url)
        self.wit = WikiImageCache(lang, wit_path)

    def score(self, sample: InputSample, sample_embeddings: np.ndarray, model: CLIP):
        pages: Dict = self.index.search(sample.context)
        if len(pages.get("results")) == 0:
            pages = self.index.search(sample.word)
        sample.explain("wiki", pages=[p["title"] for p in pages.get("results")])
        all_probs = []
        for page in pages.get("results"):
            title = page["title"]
            wip = self.wit.find_images(title, model)
            if wip is None: continue
            probs = self._score(sample_embeddings, wip, model)
            if probs is not None: all_probs.append(probs)
        if len(all_probs) != 0:
            results = np.vstack(all_probs)
            return results.max(axis=0)
        else:
            return np.zeros((10,))

    def _score(self, sample_embeddings: np.ndarray, wip: WikiImagePage, model: CLIP):
        model_name = model.model_name
        shape = wip.embeddings[model_name].shape
        is_empty = len(shape) == 1 and shape[0] == 0
        if is_empty: return None
        logits = np.matmul(sample_embeddings, wip.embeddings[model_name].transpose())
        return logits.max(axis=1)

    def close(self):
        self.wit.close()


if __name__ == '__main__':
    wikipedia_dump_path = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) >= 3 else "en"
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)
    logging.root.setLevel(logging.INFO)
    wiki = WikiIndex(lang=language)
    wiki.build_index(wikipedia_dump_path)