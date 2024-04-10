import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from evaluate import *
from model import *
from pre_process_data import *
from process_data import *
from train import *

# ----------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------
valid_train_responses = ["", "t", "T", "train", "Train"]
valid_predict_responses = ["p", "P", "predict", "Predict"]

valid_require_pre_process_responses = ["", "y", "Y", "yes", "Yes"]
valid_not_require_pre_process_responses = ["n", "N", "no", "No"]

valid_is_using_custom_seed_responses = ["y", "Y", "yes", "Yes"]
valid_not_using_custom_seed_responses = ["", "n", "N", "no", "No"]

valid_is_using_train_input_weights_responses = ["y", "Y", "yes", "Yes"]
valid_not_using_train_input_weights_responses = ["", "n", "N", "no", "No"]


def main(
    train_or_predict: bool, dataset_root_dir: str, require_pre_process: bool, seed: int
):
    # ----------------------------------------------------------------------------------------------------
    # Global Seed Setting - ensure reproducibility
    # ----------------------------------------------------------------------------------------------------
    random.seed(seed)
    np.random.seed(seed)

    # Set seed for CPU and all GPUs
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN convolution operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ----------------------------------------------------------------------------------------------------
    # Pre-process Data - extract frames and faces from the dataset root directory
    # ----------------------------------------------------------------------------------------------------
    extract_frames_and_faces(dataset_root_dir)

    # ----------------------------------------------------------------------------------------------------
    # Process Data - create data loaders from the pre-processed data
    # ----------------------------------------------------------------------------------------------------
    train_loader, test_loader, val_loader = get_data_loaders(dataset_root_dir, seed)

    # ----------------------------------------------------------------------------------------------------
    # Model - initialise the model
    # ----------------------------------------------------------------------------------------------------
    model = get_model()

    # ----------------------------------------------------------------------------------------------------
    # Train - train the model
    # ----------------------------------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    # L2 Regularization
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = train_model(model, criterion, optimizer, train_loader, num_epochs=10)

    # Save weights
    torch.save(model.state_dict(), "weights/BCE_LR1e-3_EPOCH10.pth")

    # ----------------------------------------------------------------------------------------------------
    # Evaluate
    # ----------------------------------------------------------------------------------------------------
    # Load weights
    # model.load_state_dict(torch.load('weights/BCE_LR0.001_EPOCH10.pth'))

    # Evaluate the model
    validate(model, criterion, val_loader)
    evaluate(model, test_loader)

    val_accuracy, val_precision, val_recall, val_f1 = evaluate(model, val_loader)
    print(
        f"Validation - Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}"
    )

    test_accuracy, test_precision, test_recall, test_f1 = evaluate(model, test_loader)
    print(
        f"Test - Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1 Score: {test_f1:.4f}"
    )


def prompt_train_or_predict() -> bool:
    train_or_predict_model_input = "!invalid!"

    while (
        train_or_predict_model_input not in valid_train_responses
        and train_or_predict_model_input not in valid_predict_responses
    ):
        train_or_predict_model_input = input(
            "Do you want to train the model or make predictions (T/P)? "
        )

        if train_or_predict_model_input == "":
            print("Default selection: Train")

        if train_or_predict_model_input in valid_train_responses:
            train_or_predict = True
        elif train_or_predict_model_input in valid_predict_responses:
            train_or_predict = False
        else:
            print(
                'Invalid response. Expected input: "", "t", "T", "train", "Train", "p", "P", "predict", or "Predict".\n'
            )

    return train_or_predict


def prompt_dataset_root_dir() -> str:
    dataset_root_dir = "!invalid!"

    while not (os.path.exists(dataset_root_dir)):
        dataset_root_dir = input(
            "Provide the relative path to your dataset root directory: "
        )

        if not (os.path.exists(dataset_root_dir)):
            print(
                f'Invalid dataset root directory path. Directory "{dataset_root_dir}" not found.\n'
            )
        else:
            if not (os.path.exists(f"{dataset_root_dir}/real")):
                print(
                    f'Invalid dataset root directory path. Subdirectory "{dataset_root_dir}/real" not found.'
                )
            if not (os.path.exists(f"{dataset_root_dir}/fake")):
                print(
                    f'Invalid dataset root directory path. Subdirectory "{dataset_root_dir}/fake" not found.'
                )
            if not (os.path.exists(f"{dataset_root_dir}/real")) or not (
                os.path.exists(f"{dataset_root_dir}/fake")
            ):
                print()
                dataset_root_dir = "!invalid!"

    return dataset_root_dir


def prompt_require_pre_process(dataset_root_dir: str) -> bool:
    require_pre_process = False

    if not (os.path.exists(f"{dataset_root_dir}/real_faces")):
        print(
            f'Dataset subdirectory "{dataset_root_dir}/real_faces" not found. Pre-processing of data is required.'
        )
        require_pre_process = True

    if not (os.path.exists(f"{dataset_root_dir}/fake_faces")):
        print(
            f'Dataset subdirectory "{dataset_root_dir}/fake_faces" not found. Pre-processing of data is required.'
        )
        require_pre_process = True

    if require_pre_process:
        return True

    require_pre_process_input = "!invalid!"

    while (
        require_pre_process_input not in valid_require_pre_process_responses
        and require_pre_process_input not in valid_not_require_pre_process_responses
    ):
        require_pre_process_input = input(
            "Do you need to pre-process the dataset (Y/N)? "
        )

        if require_pre_process_input == "":
            print("Default selection: Yes")

        if require_pre_process_input in valid_require_pre_process_responses:
            require_pre_process = True
        elif require_pre_process_input in valid_not_require_pre_process_responses:
            require_pre_process = False
        else:
            print(
                'Invalid response. Expected input: "", "y", "Y", "yes", "Yes", "n", "N", "no", or "No".\n'
            )

    return require_pre_process


def prompt_weights_path(train_or_predict: bool) -> str:
    weights_path = "!invalid!"
    weights_dir = "!invalid!"
    weights_filename = "!invalid!"

    # If the user is training the model, ask for relative path to where the user wants to save the weights
    if train_or_predict:
        while not (os.path.exists(weights_dir)):
            weights_dir = input("Provide the relative path to your weights directory: ")

            if not (os.path.exists(weights_dir)):
                print("Invalid weights directory path. Path not found.\n")

        weights_path = f"{weights_dir}/{weights_filename}"

        while not (os.path.exists(weights_path)):
            weights_filename = input("Provide the name of the weights output file: ")
            weights_path = f"{weights_dir}/{weights_filename}"

            try:
                with open(weights_path, "w") as weights_file:
                    pass
            except:
                print("Error: could not create the weights file.\n")

            if not (os.path.exists(weights_path)):
                print("Error: could not create the weights file.\n")

    # Else if the user is using the model to predict, ask for relative path to the input weights file
    else:
        while not (os.path.exists(weights_path)):
            weights_path = input(
                "Provide the relative path to your input weights file: "
            )

            if not (os.path.exists(weights_path)):
                print("Invalid weights file path. File not found.\n")

    return weights_path


def prompt_seed() -> int:
    is_using_custom_seed_input = "!invalid!"

    while (
        is_using_custom_seed_input not in valid_is_using_custom_seed_responses
        and is_using_custom_seed_input not in valid_not_using_custom_seed_responses
    ):
        is_using_custom_seed_input = input("Do you wish to use a custom seed (Y/N)? ")

        if is_using_custom_seed_input == "":
            print("Default selection: No")

        if is_using_custom_seed_input in valid_is_using_custom_seed_responses:
            is_using_custom_seed = True
        elif is_using_custom_seed_input in valid_not_using_custom_seed_responses:
            is_using_custom_seed = False
        else:
            print(
                'Invalid response. Expected input: "", "y", "Y", "yes", "Yes", "n", "N", "no", or "No".\n'
            )

    if is_using_custom_seed:
        custom_seed_input = "!invalid!"

        while not (custom_seed_input.isnumeric()):
            custom_seed_input = input("Provide the seed (int): ")

            if not (custom_seed_input.isnumeric()):
                print("Invalid response. Expected input: integer.\n")

        custom_seed = int(custom_seed_input)
    else:
        custom_seed = 33

    return custom_seed


def prompt_train_input_weights_path() -> str:
    is_using_train_input_weights_input = "!invalid!"

    while (
        is_using_train_input_weights_input
        not in valid_is_using_train_input_weights_responses
        and is_using_train_input_weights_input
        not in valid_not_using_train_input_weights_responses
    ):
        is_using_train_input_weights_input = input(
            "Do you want to start training the model from previously saved weights (Y/N)? "
        )

        if is_using_train_input_weights_input == "":
            print("Default selection: No")

        if (
            is_using_train_input_weights_input
            in valid_is_using_train_input_weights_responses
        ):
            is_using_train_input_weights = True
        elif (
            is_using_train_input_weights_input
            in valid_not_using_train_input_weights_responses
        ):
            is_using_train_input_weights = False
        else:
            print(
                'Invalid response. Expected input: "", "y", "Y", "yes", "Yes", "n", "N", "no", or "No".\n'
            )

    if is_using_train_input_weights:
        train_input_weights_path = "!invalid!"

        # Ask for relative path to the input weights file
        while not (os.path.exists(train_input_weights_path)):
            train_input_weights_path = input(
                "Provide the relative path to your input weights file: "
            )

            if not (os.path.exists(train_input_weights_path)):
                print("Invalid weights file path. File not found.\n")
    else:
        train_input_weights_path = False

    return train_input_weights_path


if __name__ == "__main__":
    train_or_predict = False  # True -> train, False -> predict
    dataset_root_dir = ""
    require_pre_process = False
    weights_path = ""
    seed = 33
    train_input_weights_path = False

    print()

    # If user did not provide any arguments, ask if the user wants to train the model
    if len(sys.argv) == 1:
        train_or_predict = prompt_train_or_predict()
        print()

    if len(sys.argv) == 2:
        if sys.argv[1] in valid_train_responses:
            train_or_predict = True
        elif sys.argv[1] in valid_predict_responses:
            train_or_predict = False
        else:
            print(
                'Invalid argument for "train_or_predict". Expected input: "", "t", "T", "train", "Train", "p", "P", "predict", or "Predict".\n'
            )
            train_or_predict = prompt_train_or_predict()
            print()

    # If user provided "train_or_predict" or no arguments, ask for the dataset root directory
    if len(sys.argv) <= 2:
        dataset_root_dir = prompt_dataset_root_dir()
        print()

    if len(sys.argv) == 3:
        dataset_root_dir = sys.argv[2]

        if not (os.path.exists(f"{dataset_root_dir}/real")):
            print(
                f'Invalid argument for "dataset_root_dir". Subdirectory "{dataset_root_dir}/real" not found.'
            )
            require_pre_process = True

        if not (os.path.exists(f"{dataset_root_dir}/fake")):
            print(
                f'Invalid argument for "dataset_root_dir". Subdirectory "{dataset_root_dir}/fake" not found.'
            )
            require_pre_process = True

        if not (os.path.exists(f"{dataset_root_dir}/real")) or not (
            os.path.exists(f"{dataset_root_dir}/fake")
        ):
            print()
            dataset_root_dir = prompt_dataset_root_dir()
            print()

    # If the user provided "train_or_predict" and "dataset_root_dir" or fewer arguments, ask if the user requires
    # pre-processing of data
    if len(sys.argv) <= 3:
        require_pre_process = prompt_require_pre_process(dataset_root_dir)
        print()

    if len(sys.argv) == 4:
        if sys.argv[3] in valid_require_pre_process_responses:
            require_pre_process = True
        elif sys.argv[3] in valid_not_require_pre_process_responses:
            require_pre_process = False
        else:
            print(
                'Invalid argument for "require_pre_process" argument. Expected input: "", "y", "Y", "yes", "Yes", "n", "N", "no", or "No".\n'
            )
            require_pre_process = prompt_require_pre_process(dataset_root_dir)
            print()

        if not (os.path.exists(f"{dataset_root_dir}/real_faces")):
            print(
                f'Dataset subdirectory "{dataset_root_dir}/real_faces" not found. Pre-processing of data is required.'
            )
            require_pre_process = True

        if not (os.path.exists(f"{dataset_root_dir}/fake_faces")):
            print(
                f'Dataset subdirectory "{dataset_root_dir}/fake_faces" not found. Pre-processing of data is required.'
            )
            require_pre_process = True

    # If the user provided "train_or_predict", "dataset_root_dir" and "require_pre_process" or fewer arguments, ask for
    # the weights file path
    if len(sys.argv) <= 4:
        weights_path = prompt_weights_path(train_or_predict)
        print()

    if len(sys.argv) == 5:
        weights_path = sys.argv[4]
        if not (os.path.exists(weights_path)):
            print(f'Invalid argument for "weights_path". Weights file not found.\n')
            weights_path = prompt_weights_path(train_or_predict)
            print()

    # If the user provided "train_or_predict", "dataset_root_dir", "require_pre_process" and "weights_path" or fewer
    # arguments, ask for the seed value
    if len(sys.argv) <= 5:
        seed = prompt_seed()
        print()

    if len(sys.argv) == 6:
        if not (sys.argv[5].isnumeric()):
            print(f'Invalid argument for "seed". Expected input: integer.\n')
            seed = prompt_seed()
            print()
        else:
            seed = int(sys.argv[5])

    # If the user provided "train_or_predict = train", "dataset_root_dir", "require_pre_process", "weights_path" and
    # "seed" or fewer arguments, ask if the user wants to start training using previously saved weights
    if len(sys.argv) <= 6:
        if train_or_predict:
            train_input_weights_path = prompt_train_input_weights_path()
            print()
        else:
            train_input_weights_path = False

    if len(sys.argv) == 7:
        train_input_weights_path = int(sys.argv[6])

        if not (os.path.exists(train_input_weights_path)):
            print(
                f'Invalid argument for "train_input_weights_path". Weights file not found.\n'
            )
            train_input_weights_path = prompt_train_input_weights_path()
            print()

    if len(sys.argv) <= 7:
        print("Selected parameters:")
        print("    train_or_predict:", train_or_predict)
        print("    dataset_root_dir:", dataset_root_dir)
        print("    require_pre_process:", require_pre_process)
        print("    weights_path:", weights_path)
        print("    seed:", seed)
        print("    train_input_weights_path:", train_input_weights_path)
        print()

        # main(
        #     train_or_predict,
        #     dataset_root_dir,
        #     require_pre_process,
        #     weights_path,
        #     seed,
        #     train_input_weights_path,  # Used as weights input for "train mode". Ignored for "predict" mode.
        # )
    else:
        print("Invalid usage.")
