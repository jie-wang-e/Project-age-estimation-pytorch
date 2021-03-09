import argparse
import better_exceptions
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset
from imgaug import augmenters as iaa


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.OneOf([
                iaa.Sometimes(0.25, iaa.AdditiveGaussianNoise(scale=0.1 * 255)),
                iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0)))
            ]),
            iaa.Affine(
                rotate=(-20, 20), mode="edge",
                scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}
            ),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
            iaa.GammaContrast((0.3, 2)),
            iaa.Fliplr(0.5),
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.aug.augment_image(img)
        return img


class FaceDataset(Dataset):
    def __init__(self, data_dir_age, data_dir_extended, data_type, img_size=224, augment=False, age_stddev=1.0):
        assert (data_type in ("train", "valid", "test"))
        csv_path_age = Path(data_dir_age).joinpath(f"gt_avg_{data_type}.csv")
        csv_path_extended = Path(data_dir_extended).joinpath(f"allcategories_{data_type}.csv")
        img_dir = Path(data_dir_age).joinpath(data_type)

        self.img_size = img_size
        self.augment = augment
        self.age_stddev = age_stddev

        if augment:
            self.transform = ImgAugTransform()
        else:
            self.transform = lambda i: i

        self.x = []
        self.y = []
        self.std = []
        self.gender = []
        self.race = []
        self.makeup = []
        self.time = []
        self.happiness = []
        df_age = pd.read_csv(str(csv_path_age))
        df_extended = pd.read_csv(str(csv_path_extended))
        ignore_path = Path(__file__).resolve().parent.joinpath("ignore_list.csv")
        ignore_img_names = list(pd.read_csv(str(ignore_path))["img_name"].values)

        for _, row in df_age.iterrows():
            img_name = row["file_name"]

            if img_name in ignore_img_names:
                continue

            img_path = img_dir.joinpath(img_name + "_face.jpg")
            assert (img_path.is_file())
            self.x.append(str(img_path))
            self.y.append(row["apparent_age_avg"])
            self.std.append(row["apparent_age_std"])

        for _, row in df_extended.iterrows():
            img_name = row["file"]

            if img_name in ignore_img_names:
                continue

            self.gender.append(row["gender"])
            # self.race.append(row["race"])
            # self.makeup.append(row["makeup"])
            # self.time.append(row["time"])
            # self.happiness.append(row["happiness"])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img_path = self.x[idx]
        age = self.y[idx]
        gender = self.gender[idx]

        if self.augment:
            age += np.random.randn() * self.std[idx] * self.age_stddev

        img = cv2.imread(str(img_path), 1)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = self.transform(img).astype(np.float32)
        return torch.from_numpy(np.transpose(img, (2, 0, 1))), np.clip(round(age), 0, 100), int(gender == "male")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir_age", type=str, required=True)
    parser.add_argument("--data_dir_extended", type=str, required=True)
    args = parser.parse_args()
    dataset = FaceDataset(args.data_dir_age, args.data_dir_extended, "train")
    print("train dataset len: {}".format(len(dataset)))
    dataset = FaceDataset(args.data_dir_age, args.data_dir_extended, "valid")
    print("valid dataset len: {}".format(len(dataset)))
    dataset = FaceDataset(args.data_dir_age, args.data_dir_extended, "test")
    print("test dataset len: {}".format(len(dataset)))


if __name__ == '__main__':
    main()

