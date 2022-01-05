# OCR-Pipeline-with-Keras

The keras-ocr package generally consists of two parts: a `Detector` and a `Recognizer`:

* `Detector` is responsible for creating bounding boxes for the words of the text.
* `Recognizer` is responsible for processing batch of cropped parts of the initial image.

Keras-ocr connects this two parts into seamless pipeline. "Out of the box", it can handle a wide range of images with texts. But in a specific task, when the field of possible images with texts is greatly narrowed, it shows itself badly in the `Recognizer` part of the task.


In this regard, the task of fine-tuning `Recognizer` on a custom dataset was set.

---

### Virtual environment and packages

```bash
$ python3 -m venv keras_ocr
$ pip install keras-ocr
```

And TRDG library for synthetic text generation.

```bash
$ pip install trdg
```

---

### Synthetic data generation

We will use the TRDG library to generate synthetic text. All necessary code presented in the `data_generation.py`. Things you need to know:

* You choose template for generating text, e.g. if template is `"({}{}/{})"`, then all brackets will be randomly filled with symbols from `alphabet`. You need to specify your own instance of `StringTemplate` classs.

* You choose the `alphabet`. In our example case it contains only digits. P.S. Some of the repeated in `data_generation.py`, hence emperical distribution probability for each symbol defined as fraction of n_repeats to alphabet_size.

* You can choose your own fonts. To do this, follow instruction:
    1. Download needed fonts as `.ttf` files
    2. Go to `trdg` fonts directory `./keras_ocr/lib/python3.8/site-packages/trdg/fonts/`
    3. Create directory `$ mkdir cs` (cs means custom fonts), you can chooce the disered name
    4. Place fonts files in this dir
    5. (For Mac users only) Don't forget to remove `.DS_Store` from this folder


* You can chooce image  background for text. When creating instance of `GeneratorFromStrings` in function `generate_data_units(...)`, provide folder with images with arg `image_dir`

##### High-level API in the `data_generation.py`

```pyhton
data_generator = DataGenerator(string_templates=[StringTemplate('{}{}{}{}{}{}{}', 7)])

data_generator.generate(n_patches=20000, n_total_samples=550, path='DigitsBracketsDataset/train')
```

* `n_patches` -- number of different strings from provided template
* `n_total_samples` -- number of total samples from patches
* `path` -- dir to save samples


---

### Fine tuning Recognizer
Follow instruction in `fine_tuning.ipynb`. Don't forget to add function `get_custom_dataset(...)` to `datasets.py` in keras-ocr package directory (`./keras_ocr/lib/python3.8/site-packages/keras_ocr/datasets.py`):

```python
def get_custom_dataset(path: str, split: str):
    """
    param: path: path to dataset root dir (include train/test dirs)
    Returns:
        A recognition dataset as a list of (filepath, box, word) tuples
    """
    data = []
    if split == 'train':
        train_dir = os.path.join(path, 'train')
        data.extend(
            _read_born_digital_labels_file(
                labels_filepath=os.path.join(train_dir, "gt.txt"),
                image_folder=train_dir,
            )
        )
    elif split == 'test':
        test_dir = os.path.join(path, 'test')
        data.extend(
            _read_born_digital_labels_file(
                labels_filepath=os.path.join(test_dir, 'gt.txt'), 
                image_folder=test_dir
            )
        )
    return data 
```
