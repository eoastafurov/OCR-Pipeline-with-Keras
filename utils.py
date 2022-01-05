import numpy as np
import json

class ImageUtils:
    def __init__(self, borders: () = None):
        if borders is None:
            borders = (180, 550, 280, 450)
        self.borders = borders

    def crop_significant(self, image: np.ndarray) -> np.ndarray:
        return image[
               self.borders[0]:self.borders[1],
               self.borders[2]:self.borders[3]
               ]


class PredictionsSingle:
    def __init__(
            self,
            raw_predictions: np.ndarray,
            MASK: str = None
    ):
        if MASK is None:
            MASK = '010'
        self.MASK = MASK
        self.raw_predictions = raw_predictions
        self.predictions = self.convert_preds_to_ids_(raw_predictions)

    @staticmethod
    def throw_bboxes_(raw_predictions: np.ndarray) -> np.ndarray:
        n_axis = len(raw_predictions.shape)
        if n_axis != 2:
            raise ValueError('Incorrect axis number: expected 2, received {}'.format(n_axis))

        return raw_predictions[:, 0]

    def convert_preds_to_ids_(self, raw_predictions: np.ndarray) -> dict:
        raw_predictions = self.throw_bboxes_(raw_predictions)

        n_axis = len(raw_predictions.shape)
        if n_axis != 1:
            raise ValueError('Incorrect axis number: expected 1, received {}'.format(n_axis))

        n_preds = raw_predictions.shape[0]

        if n_preds % len(self.MASK) != 0:
            raise ValueError(
                'Number of predictions not suited for mask: n_preds = {}, \
                len(mask) = {}'.format(n_preds, len(self.MASK))
            )

        full_mask = np.array(
            [int(i) for i in self.MASK * (n_preds // len(self.MASK))], dtype=np.bool8
        )

        ids_dict = {}
        for i, elem in enumerate(raw_predictions[np.where(full_mask)]):
            ids_dict[i] = elem

        return ids_dict

    def __str__(self):
        return str(self.predictions)

    def dump_to_json(self, path: str = './ids.json') -> None:
        with open(path, 'w') as f:
            json.dump(self.predictions, f)
