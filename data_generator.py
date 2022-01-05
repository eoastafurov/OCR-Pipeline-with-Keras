from typing import List, Optional
import numpy as np
from trdg.generators import (
    GeneratorFromStrings,
)
import cv2

# pipeline = keras_ocr.pipeline.Pipeline()


class StringTemplate:
    """
    e.g.    template = '{}{}{}{}'
            n_symbols = 4
    """
    def __init__(self, template: str, n_symbols: int):
        self.template = template
        self.n_symbols = n_symbols


class DataUnit:
    def __init__(self, image: np.array, label: str):
        self.image = image
        self.label = label
        # self.crop_()

    # def crop_(self):
    #     prediction_groups = pipeline.recognize([self.image])
    #     for i, e in enumerate(prediction_groups[0]):
    #         coords = e[1]
    #         # print(coords[0][1],coords[3][1], coords[0][0],coords[1][0])
    #         cropped = self.image[int(coords[0][1]):int(coords[3][1]), int(coords[0][0]):int(coords[1][0])]
    #     self.image = cropped


class DataGenerator:
    def __init__(
            self,
            alphabet=None,
            string_templates=None
    ):
        if string_templates is None:
            string_templates = [StringTemplate('({}/{}{})', 3), StringTemplate('{}{}{}', 3)]
        if alphabet is None:
            alphabet = [
                '0', '0', '1', '2', '2', '3', '3',
                '4', '5', '6', '7', '8', '8', '9'
            ]
        self.alphabet = alphabet
        self.string_templates = string_templates

    def generate_patches_(self, n_patches: int = 10) -> List[str]:
        patches = []
        for template in self.string_templates:
            for _ in range(n_patches):
                symbols = []
                for _ in range(template.n_symbols):
                    symbols.append(np.random.choice(self.alphabet))
                patches.append(template.template.format(*symbols))
        np.random.shuffle(patches)
        return patches

    def generate_data_units(self, n_patches: int, n_total_samples: int) -> List[DataUnit]:
        data = []
        patches = self.generate_patches_(n_patches)
        generator = GeneratorFromStrings(
            strings=patches,
            count=n_total_samples,
            # blur=0.2,
            random_blur=True,
            image_dir='/Users/evgenijastafurov/Desktop/2021/OCR_PYTHON.nosync/TRDG/images',
            background_type=3,
            language='cs2',
            text_color='#2f3a4d',
            size=15,
            margins=(1, 1, 2, 1)
        )
        assert os.path.exists(generator.image_dir)

        while True:
            try:
                img, lbl = generator.next()
                data.append(DataUnit(np.array(img), lbl))
            except StopIteration:
                break

        return data

    def generate(self, n_patches: int, n_total_samples: int, path: str) -> None:
        data = self.generate_data_units(n_patches, n_total_samples)
        gt_labels = []
        data_format = 'png'
        for i, data_unit in enumerate(data):
            filename = '{}/{}.{}'.format(path, i, data_format)
            label = data_unit.label
            img = data_unit.image
            # gt_labels.append('{}, \"{}\"'.format(filename, label))
            gt_labels.append('{}, \"{}\"'.format('{}.{}'.format(i, data_format), label))
            cv2.imwrite(filename, img)

        with open('{}/gt.txt'.format(path), 'w') as f:
            f.write("\n".join(gt_labels))

        print('Done!')
        return


data_generator = DataGenerator(string_templates=[StringTemplate('{}{}{}{}{}{}{}', 7)])
# data_generator.generate_patches_()

data_generator.generate(n_patches=20000, n_total_samples=550, path='DigitsBracketsDataset/train')
data_generator.generate(n_patches=20000, n_total_samples=550, path='DigitsBracketsDataset/test')
