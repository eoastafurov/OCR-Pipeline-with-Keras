import numpy as np
from typing import List, Optional
import tensorflow as tf
import keras_ocr

from utils import PredictionsSingle, ImageUtils

IMAGE_SHAPE = (700, 1366, 3)


class NumberOCR:
    def __init__(
            self,
            weights_path: str,
            alphabet: Optional[List] = None
    ):
        if alphabet is None:
            alphabet = [str(i) for i in range(10)]

        # self.recognizer = keras_ocr.recognition.Recognizer(alphabet=alphabet)
        self.iu = ImageUtils()
        self.recognizer = keras_ocr.recognition.Recognizer()
        self.load_weights_(self.recognizer.model, weights_path)
        self.pipeline = keras_ocr.pipeline.Pipeline(
            detector=keras_ocr.detection.Detector(),
            recognizer=self.recognizer
        )

    @staticmethod
    def load_weights_(model: tf.keras.Model, weights_path: str) -> None:
        model.load_weights(weights_path)

    def predict(self, image: np.ndarray) -> PredictionsSingle:
        if image.shape != IMAGE_SHAPE:
            raise ValueError(
                'Incorrect input image shape: expected {}, \
                received {}'.format(IMAGE_SHAPE, image.shape)
            )
        image_batched = np.expand_dims(
            self.iu.crop_significant(image),
            axis=0
        )
        raw_predictions = np.squeeze(
            self.pipeline.recognize(image_batched),
            axis=0
        )
        return PredictionsSingle(raw_predictions)
