import cv2
from model import NumberOCR
import pytest


def test():
    img = cv2.imread('./tests/1.png')  # full real screenshot
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    nocr = NumberOCR(weights_path='./weights/fine_tune_loss4_small_font.h5')

    predictions = nocr.predict(img)

    assert str(predictions) == "{0: '5455087', 1: '5455088', 2: '5455089', 3: '5455082', 4: '5455083'}"

# fig, ax = plt.subplots(nrows=1, figsize=(20, 20))
# keras_ocr.tools.drawAnnotations(image=img, predictions=prediction_groups, ax=ax)