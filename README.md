## Getting started

### 1. Download the VWSD task data

Download the trial, train and test sets from the [task page](https://raganato.github.io/vwsd/). 
Place the data in the same directory, in subdirectories named `trial_v1`, `train_v1`, `test_v1`, respectively.
In each subdirectory, create a folder with images for the specific subset named `trial_images_v1`, `train_images_v1`, or `test_images_v1`.

### 2. Prepare Wikipedia index

Wikipedia retrieval is handled by `wiki-index` application available in a separate repository: https://github.com/sdadas/wiki-index.
Clone the repository, execute `mvn package` to build the app, and `java -jar target/wiki-index.jar` to run it.
In order to build a new index, you need to download the appropriate Wikipedia dump in `pages-articles` format from the [Wikimedia Downloads](https://dumps.wikimedia.org/) respository.
Then, you can execute the `vwsd/wikipedia.py` script.

Instead of building an index from scratch, you can also download our [pre-built indexes](https://1drv.ms/u/c/71e57b049afd0d97/EYsDLJf-ykRKjsBsIyS-gvAB1RmvbFe_LumC1xPJWubESw?e=N2ptLe) for English, Italian and Persian.
Unzip the archive to the directory from which you run the Java program.

### 3. Download WIT dataset

Download WIT dataset from the [official repository](https://github.com/google-research-datasets/wit/blob/main/DATA.md).
You should download all `*.tsv.gz` files from the training, test and validation parts of the dataset, then unpack them to the directory of your choice.

### 4. Run the code for English

To generate predictions for the test dataset, execute the following command:

```shell
python run_model.py --wit_dir [path_to_wit_directory] --data_dir [path_to_vwsd_task_data] --data_split test --lang en
```

### 5. Download additional models for Italian and Persian

For languages other than English, the code uses additional models which need to be downloaded. 
First, download the fine-tuned [CLIP text encoders for Italian and Persian](https://1drv.ms/u/c/71e57b049afd0d97/EWBLYUjSANhAtauVGgh-ULgBJyQbAZXgMZBOyh1nC07f4Q?e=vPdNE5), and then extract them in the project directory.
Next, create a new directory named `embeddings` in the project root directory.
Download the [FastText models](https://fasttext.cc/docs/en/crawl-vectors.html) for Italian and Persian, and place them in the newly created directory.

### 6. Run the code for Italian or Persian

To run the code for a language other than English, execute the same command as in step 4, changing the `lang` parameter.
For example:

```shell
python run_model.py --wit_dir [path_to_wit_directory] --data_dir [path_to_vwsd_task_data] --data_split test --lang it
```
