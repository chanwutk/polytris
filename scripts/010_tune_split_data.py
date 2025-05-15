SELECTIVITY = 0.01
CLASSIFIER_SIZES = [32, 64, 128]
PADDING_SIZES = [0, 1, 2]
DIFF_THRESHOLDS = [10, 20, 30]
DIFF_SCALE = [1, 2, 4]


def parse_args():
    import argparse


    parser = argparse.ArgumentParser(description="Tune parameters for the model.")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="The model to use for tuning parameters.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset.json",
        help="The dataset to use for tuning parameters.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="The directory to save the tuned parameters.",
    )
    return parser.parse_args()


# todo: cut snippet of videos for tuning
# todo: run detector
# todo: split dataset
# todo: train classifier
