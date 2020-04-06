import argparse
import os

import numpy as np
import pandas as pd
from pathlib import Path

# -- class(label) info
class_file = open("patent_image/wnids.txt", "r")
classes = [x.strip() for x in class_file.readlines()]
class_file.close()

# -- path info
TRAIN_PATH = "patent_image/design/"
TRIPLET_PATH = "design_triplet.csv"


def list_pictures(directory):
    return [Path(root) / f
            for root, _, files in os.walk(directory) for f in files]


def get_negative_images(all_images, image_names, num_neg_images):
    """
    Get out class images
    """
    random_numbers = np.arange(len(all_images))
    np.random.shuffle(random_numbers)
    if int(num_neg_images) > (len(all_images) - 1):
        num_neg_images = len(all_images) - 1
    neg_count = 0
    negative_images = []
    for random_number in list(random_numbers):
        if all_images[random_number] not in image_names:
            negative_images.append(all_images[random_number])
            neg_count += 1
            if neg_count > (int(num_neg_images) - 1):
                break

    return negative_images


def get_positive_images(image_name, image_names, num_pos_images):
    """
    Get in class images
    """
    random_numbers = np.arange(len(image_names))
    np.random.shuffle(random_numbers)
    if int(num_pos_images) > (len(image_names) - 1):
        num_pos_images = len(image_names) - 1
    pos_count = 0
    positive_images = []
    for random_number in list(random_numbers):
        if image_names[random_number] != image_name:
            positive_images.append(image_names[random_number])
            pos_count += 1
            if int(pos_count) > (int(num_pos_images) - 1):
                break

    return positive_images


def generate_triplets(dataset_path, num_neg_images, num_pos_images):
    """
    Generate pre-sampled triplet dataset

    Parameters
    ----------
    training: 0/1 based on training testing ot testing data
    dataset_path: path to dataset
    num_neg_images: number of negative images per query image
    num_pos_images: number of positive images per query image

    Returns
    -------
    Void, setups triplet dataset in .csv file
    """
    triplet_df = pd.DataFrame(columns=["query", "positive", "negative"])

    all_images = []
    for class_ in classes:
        all_images += list_pictures(os.path.join(dataset_path, class_))

    for class_ in classes:
        image_names = list_pictures(os.path.join(dataset_path, class_))
        for image_name in image_names:
            query_image = image_name
            positive_images = get_positive_images(image_name, image_names, num_pos_images)
            for positive_image in positive_images:
                negative_images = get_negative_images(all_images, set(image_names), num_neg_images)
                for negative_image in negative_images:
                    row = {"query": query_image,
                           "positive": positive_image,
                           "negative": negative_image}
                    print(row)
                    triplet_df = triplet_df.append(row, ignore_index=True)

    triplet_df.to_csv(TRIPLET_PATH, index=False)
    print("Sampling done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optional app description')

    parser.add_argument('--n_pos',
                        help='A number of Positive images per Query image')

    parser.add_argument('--n_neg',
                        help='A number of Negative images per Query image')

    args = parser.parse_args()

    if int(args.n_neg) < 1 or int(args.n_pos) < 1:
        print('Number of Negative(Positive) Images cannot be less than 1...')
        quit()

    dataset_path = TRAIN_PATH
    print("Grabbing images from: " + dataset_path)
    print("Number of Positive image per Query image: " + args.n_pos)
    print("Number of Negative image per Query image: " + args.n_neg)

    generate_triplets(dataset_path, args.n_neg, args.n_pos)
