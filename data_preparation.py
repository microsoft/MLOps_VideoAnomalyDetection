"""
This file is meant to be run as a step in an AML pipline created by
pipelines_slave.py, but it can also be run independently.

It preprocesses video frames (stored as individual image files) so
that they can be use for training a prednet architecture

"""

import os
import numpy as np
import hickle as hkl
import glob
from sklearn.model_selection import train_test_split
from PIL import Image
import argparse


def main():
    """ Create image datasets. Processes images and saves them in train,
    val, test splits. For each split, this creates a numpy array w/
    dimensions n_images, height, width, depth.
    """

    # define input arguments that this script accepts
    parser = argparse.ArgumentParser(description="Process input arguments")
    parser.add_argument(
        "--raw_data",
        default="./data/UCSD_Anomaly_Dataset.v1p2/",
        type=str,
        dest="raw_data",
        help="data folder mounting point",
    )
    parser.add_argument(
        "--preprocessed_data",
        default="./data/preprocessed/",
        type=str,
        dest="preprocessed_data",
        help="data folder mounting point",
    )
    parser.add_argument(
        "--n_frames",
        default=200,
        type=int,
        dest="n_frames",
        help="length of video sequences in input data",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset",
        default="UCSDped1",
        help="the dataset that we are using",
        type=str,
        required=False,
    )

    test_size = 0.5
    # process input arguments
    args = parser.parse_args()
    raw_data = os.path.join(args.raw_data, args.dataset)
    preprocessed_data_path = os.path.join(args.preprocessed_data, args.dataset)
    # dataset = os.path.basename(raw_data)
    assert args.dataset in ["UCSDped1", "UCSDped2"], (
        "Dataset (%s) not valid." % args.dataset
    )
    if not (preprocessed_data_path is None):
        os.makedirs(preprocessed_data_path, exist_ok=True)
        print("%s created" % preprocessed_data_path)

    # orig image size in USCD data: h,w = (158, 238)
    desired_im_sz = (152, 232)
    skip_frames = 0

    print("Input data:", raw_data)
    # Recordings used for training and validation
    # recordings_parent_folder = os.path.join(raw_data, folders[0])
    recordings = glob.glob(os.path.join(raw_data, "Train", "Train*[0-9]"))
    recordings = sorted(recordings)
    n_recordings = len(recordings)
    print("Found %s recordings for training" % n_recordings)
    print("Folders: "),
    print(os.listdir(os.path.join(raw_data, "Train")))

    train_recordings = list(zip([raw_data] * n_recordings, recordings))

    # Recordings used for testing
    # recordings_parent_folder = os.path.join('data', folders[0])
    recordings = glob.glob(os.path.join(raw_data, "Test", "Test*[0-9]"))
    recordings = sorted(recordings)
    n_recordings = len(recordings)
    print("Found %s recordings for validation and testing" % n_recordings)
    print("Using %d percent for testing" % (test_size * 100))
    print("Folders: "),
    print(os.listdir(os.path.join(raw_data, "Test")))

    recordings = list(zip([raw_data] * n_recordings, recordings))

    # we split the training data into training and validation set randomly,
    # but with fixed random_state, for reproducability
    val_recordings, test_recordings = train_test_split(
        recordings, test_size=test_size, random_state=123
    )

    # create a dictionary of lists for train/test/val datasets
    splits = {s: [] for s in ["train", "test", "val"]}
    splits["train"] = train_recordings
    splits["val"] = val_recordings
    splits["test"] = test_recordings

    for split in splits:
        im_list = []  # list of all images of a split
        source_list = []  # corresponds to recording that image came from
        i = 0
        for _, folder in splits[split]:
            files = glob.glob(os.path.join(folder, "*.tif"), recursive=False)
            files = sorted(files)
            for skip in range(0, skip_frames + 1):
                # print(skip)
                for c, f in enumerate(files):
                    if c % (skip_frames + 1) == skip:
                        # print(c, skip, f)
                        im_list.append(f)
                        source_list.append(os.path.dirname(f))
                        i += 1

        print(
            "Creating " + split + " data set "
            "with " + str(len(im_list)) + " images"
        )

        # X is 4D w/ axes: n_images, height, width, depth (e.g. rgb, grayscale)
        X = np.zeros((len(im_list),) + desired_im_sz + (3,), np.uint8)
        for i, im_file in enumerate(im_list):
            try:
                im = Image.open(im_file).convert(mode="RGB")
            except Exception as e:
                print(e)
                print(im_file)
                print(
                    "something with this file. You can open and investigate"
                    " manually. It's probably OK to ignore, unless you get"
                    "a ton of these warnings."
                )
            try:
                # scale and crop image
                X[i] = np.asarray(process_im(im, desired_im_sz))
            except Exception as e:
                print(e)
                print(im_file)
                raise

        if split in ["val", "test"]:
            print("Creating anomaly dataset for %s split" % split)
            # The next step is to merge that with the labels for
            # anomalies in the ucsd dataset
            # UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/UCSDped1.m
            anom_anot_filename = os.path.join(
                raw_data, "Test", "%s.m" % args.dataset
            )
            with open(anom_anot_filename, "r") as f:
                lines = f.readlines()
            del lines[0]  # remove file header

            # extract the beginning and end of subsequences that contain
            # anomalies
            anom_indices = []
            for l, line in enumerate(lines):
                line = line.replace(":", ",")
                anom_index = line.split("[")[1].split("]")[0].split(",")
                anom_indices.append(
                    anom_index
                )

            anoms = np.zeros((X.shape[0]))

            for f, folder in enumerate(splits[split]):
                row = int(os.path.basename(folder[1])[-3:])
                anom = anom_indices[row - 1]
                while len(anom) > 0:
                    first_frame = int(anom.pop(0)) + row * args.n_frames
                    last_frame = int(anom.pop(0)) + row * args.n_frames
                    anoms[first_frame:last_frame] = 1

                    hkl.dump(
                        anoms,
                        os.path.join(
                            preprocessed_data_path,
                            "y_" + split + ".hkl"))

        # save all the data one split in one giant archive
        hkl.dump(
            X,
            os.path.join(
                preprocessed_data_path,
                "X_" + split + ".hkl"))
        hkl.dump(
            source_list,
            os.path.join(
                preprocessed_data_path,
                "sources_" + split + ".hkl"))


def process_im(im, desired_im_sz):
    """resize Image

    Arguments:
        im {[PIL.Image.Image]} -- Image to resize
    """
    im = im.resize(
        (desired_im_sz[1], desired_im_sz[0]), resample=Image.BICUBIC
    )
    return im


if __name__ == "__main__":
    main()
