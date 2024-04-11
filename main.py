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

valid_is_require_pre_process_responses = [
    "",
    "y",
    "Y",
    "yes",
    "Yes",
    "t",
    "T" "true",
    "True",
]
valid_not_require_pre_process_responses = [
    "n",
    "N",
    "no",
    "No",
    "f",
    "F",
    "false",
    "False",
]

valid_is_using_custom_seed_responses = ["y", "Y", "yes", "Yes"]
valid_not_using_custom_seed_responses = ["", "n", "N", "no", "No"]

valid_null_train_input_weights_path_responses = [
    "none",
    "None",
    "null",
    "Null",
    "false",
    "False",
]

valid_is_using_train_input_weights_responses = ["y", "Y", "yes", "Yes"]
valid_not_using_train_input_weights_responses = ["", "n", "N", "no", "No"]


def main(
    train_or_predict: str,
    dataset_root_dir: str,
    require_pre_process: bool,
    weights_path: str,
    train_input_weights_path: str = "None",
    seed: int = 33,
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
    # Get Processor Device
    # ----------------------------------------------------------------------------------------------------
    # Assign GPU as device if available, else assign cpu
    print("Cuda is available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using cuda device:", torch.cuda.get_device_name())
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # ----------------------------------------------------------------------------------------------------
    # Pre-process Data - extract frames and faces from the dataset root directory
    # ----------------------------------------------------------------------------------------------------
    if require_pre_process:
        extract_frames_and_faces(dataset_root_dir)

    # ----------------------------------------------------------------------------------------------------
    # Process Data - create data loaders from the pre-processed data
    # ----------------------------------------------------------------------------------------------------
    train_loader, test_loader, val_loader = get_data_loaders(
        dataset_root_dir, device, seed
    )

    # ----------------------------------------------------------------------------------------------------
    # Model - initialise the model
    # ----------------------------------------------------------------------------------------------------
    model = get_model(device)

    # ----------------------------------------------------------------------------------------------------
    # Train - train the model
    # ----------------------------------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    # L2 Regularization
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = train_model(model, criterion, optimizer, train_loader, 10, device)

    # Save weights
    torch.save(model.state_dict(), weights_path)

    # ----------------------------------------------------------------------------------------------------
    # Evaluate
    # ----------------------------------------------------------------------------------------------------
    # Load weights
    # model.load_state_dict(torch.load('weights/BCE_LR0.001_EPOCH10.pth'))

    # Evaluate the model
    validate(model, criterion, val_loader, device)
    evaluate(model, test_loader, device)

    val_accuracy, val_precision, val_recall, val_f1 = evaluate(
        model, val_loader, device
    )
    print(
        f"Validation - Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}"
    )

    test_accuracy, test_precision, test_recall, test_f1 = evaluate(
        model, test_loader, device
    )
    print(
        f"Test - Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1 Score: {test_f1:.4f}"
    )


def prompt_train_or_predict(train_or_predict_input="!invalid!") -> str:
    if train_or_predict_input != "!invalid!" and (
        train_or_predict_input not in valid_train_responses
        and train_or_predict_input not in valid_predict_responses
    ):
        print(
            f"Invalid value provided for 'train_or_predict': '{train_or_predict_input}'."
        )
        print(
            f"Expected input: {valid_train_responses + valid_predict_responses}",
        )
        print("Enter the correct value for the following input prompt...\n")

    while (
        train_or_predict_input not in valid_train_responses
        and train_or_predict_input not in valid_predict_responses
    ):
        train_or_predict_input = input(
            "Do you want to train the model or make predictions (T/P)? "
        )

        if (
            train_or_predict_input not in valid_train_responses
            and train_or_predict_input not in valid_predict_responses
        ):
            print(
                f"Invalid response. Expected input: {valid_train_responses + valid_predict_responses}"
            )

        print()

    if train_or_predict_input == "":
        print("Default selection: Train")

    if train_or_predict_input in valid_train_responses:
        train_or_predict = "Train"
    elif train_or_predict_input in valid_predict_responses:
        train_or_predict = "Predict"

    return train_or_predict


def prompt_dataset_root_dir(dataset_root_dir="!invalid!") -> str:
    if dataset_root_dir != "!invalid!" and not (os.path.exists(dataset_root_dir)):
        print(
            f"Invalid value provided for 'dataset_root_dir': '{dataset_root_dir}'. Directory not found."
        )
        print("Enter the correct value for the following input prompt...\n")

    while not (os.path.exists(dataset_root_dir)):
        dataset_root_dir = input(
            "Provide the relative path to the dataset root directory: "
        )

        if not (os.path.exists(dataset_root_dir)):
            print(
                f"Invalid dataset root directory path. Directory '{dataset_root_dir}' not found."
            )
        else:
            if not (os.path.exists(f"{dataset_root_dir}/real")):
                print(
                    f"Invalid dataset root directory path. Subdirectory '{dataset_root_dir}/real' not found."
                )
            if not (os.path.exists(f"{dataset_root_dir}/fake")):
                print(
                    f"Invalid dataset root directory path. Subdirectory '{dataset_root_dir}/fake' not found."
                )

        print()

    return dataset_root_dir


def prompt_require_pre_process(
    dataset_root_dir: str, require_pre_process_input="!invalid!"
) -> bool:
    if (
        not (require_pre_process_input == "!invalid!")
        and require_pre_process_input not in valid_is_require_pre_process_responses
        and require_pre_process_input not in valid_not_require_pre_process_responses
    ):
        print(
            f"Invalid value provided for 'require_pre_process': '{require_pre_process_input}'."
        )
        print(
            f"Expected input: {valid_is_require_pre_process_responses + valid_not_require_pre_process_responses}"
        )
        print("Enter the correct value for the following input prompt...\n")

    if require_pre_process_input in valid_is_require_pre_process_responses:
        require_pre_process = True
    elif require_pre_process_input in valid_not_require_pre_process_responses:
        require_pre_process = False

    while (
        require_pre_process_input not in valid_is_require_pre_process_responses
        and require_pre_process_input not in valid_not_require_pre_process_responses
    ):
        require_pre_process_input = input(
            "Do you need to perform pre-processing on the dataset (Y/N)? "
        )

        if (
            require_pre_process_input not in valid_is_require_pre_process_responses
            and require_pre_process_input not in valid_not_require_pre_process_responses
        ):
            print(
                f"Invalid response. Expected input: {valid_is_require_pre_process_responses + valid_not_require_pre_process_responses}"
            )
        else:
            if require_pre_process_input == "":
                print("Default selection: Yes")

            if require_pre_process_input in valid_is_require_pre_process_responses:
                require_pre_process = True
            elif require_pre_process_input in valid_not_require_pre_process_responses:
                require_pre_process = False

            if require_pre_process == False:
                if not (os.path.exists(f"{dataset_root_dir}/real_faces")):
                    print(
                        f"Dataset subdirectory '{dataset_root_dir}/real_faces' not found. Pre-processing of dataset will be performed."
                    )
                    require_pre_process = True

                if not (os.path.exists(f"{dataset_root_dir}/fake_faces")):
                    print(
                        f"Dataset subdirectory '{dataset_root_dir}/fake_faces' not found. Pre-processing of dataset will be performed."
                    )
                    require_pre_process = True

        print()

    return require_pre_process


def prompt_weights_path(train_or_predict: bool, weights_path="!invalid!") -> str:
    weights_dir = "/".join(weights_path.split("/")[0:-1])

    if (
        train_or_predict == "Train"
        and weights_path != "!invalid!"
        and not (os.path.exists(weights_dir))
    ):
        print(
            f"Invalid value provided for 'weights_path': '{weights_path}'. Directory '{weights_dir}' not found."
        )
        print("Enter the correct value for the following input prompt...\n")
    elif (
        train_or_predict == "Predict"
        and weights_path != "!invalid!"
        and not (os.path.exists(weights_path))
    ):
        print(
            f"Invalid value provided for 'weights_path': '{weights_path}'. File not found."
        )
        print("Enter the correct value for the following input prompt...\n")

    # If the user is training the model, ask for relative path to where the user wants to save the weights
    if train_or_predict == "Train":
        try:
            with open(weights_path, "w") as weights_file:
                pass
        except:
            print("Error: could not create the output weights file.\n")

        if not (os.path.exists(weights_path)):
            print("Error: could not create the output weights file.\n")

        while not (os.path.exists(weights_path)):
            weights_path = input(
                "Provide the relative path to the output weights file: "
            )
            weights_dir = "/".join(weights_path.split("/")[0:-1])

            if not (os.path.exists(weights_dir)):
                print(
                    f"Invalid output weights file path. Directory '{weights_dir}' not found."
                )
            else:
                try:
                    with open(weights_path, "w") as weights_file:
                        pass
                except:
                    print("Error: could not create the output weights file.")

                if not (os.path.exists(weights_path)):
                    print("Error: could not create the output weights file.")

            print()

    # Else if the user is using the model to predict, ask for relative path to the input weights file
    elif train_or_predict == "Predict":
        while not (os.path.exists(weights_path)):
            weights_path = input(
                "Provide the relative path to the input weights file: "
            )

            if not (os.path.exists(weights_path)):
                print("Invalid weights file path. File not found.")

            print()

    return weights_path


def prompt_train_input_weights_path(
    is_using_train_input_weights_input="!invalid!", train_input_weights_path="!invalid!"
) -> str:
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
                f"Invalid response. Expected input: {valid_is_using_train_input_weights_responses + valid_not_using_train_input_weights_responses}\n"
            )

    if is_using_train_input_weights:
        # Ask for relative path to the input weights file
        while not (os.path.exists(train_input_weights_path)):
            train_input_weights_path = input(
                "Provide the relative path to your input weights file: "
            )

            if not (os.path.exists(train_input_weights_path)):
                print("Invalid weights file path. File not found.\n")
    else:
        train_input_weights_path = "None"

    print()

    return train_input_weights_path


def prompt_seed(is_using_custom_seed_input="!invalid!") -> int:
    while (
        is_using_custom_seed_input not in valid_is_using_custom_seed_responses
        and is_using_custom_seed_input not in valid_not_using_custom_seed_responses
    ):
        is_using_custom_seed_input = input(
            "Do you wish to use a custom seed value (Y/N)? "
        )

        if is_using_custom_seed_input == "":
            print("Default selection: No")

        if is_using_custom_seed_input in valid_is_using_custom_seed_responses:
            is_using_custom_seed = True
        elif is_using_custom_seed_input in valid_not_using_custom_seed_responses:
            is_using_custom_seed = False
        else:
            print(
                f"Invalid response. Expected input: {valid_is_using_custom_seed_responses + valid_not_using_custom_seed_responses}.\n"
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

    print()

    return custom_seed


if __name__ == "__main__":
    train_or_predict = "train"
    dataset_root_dir = ""
    require_pre_process = False
    weights_path = ""
    train_input_weights_path = "None"
    seed = 33

    print()

    # If user did not provide all the mandatory arguments, show help message
    if len(sys.argv) < 5:
        print("Insufficient arguments provided.\n")
        print(
            "Usage: main.py <train_or_predict> <dataset_root_dir> <require_pre_process> <weights_path>[ <train_input_weights_path>][ <seed>]"
        )
        print(
            "- Angle brackets <> refer to arguments that must be replaced with the desired values."
        )
        print("- Square brackets [] refer to optional arguments.\n")

        print("Proceeding to show input prompts for the missing arguments...\n")

    # If user did not provide "train_or_predict" argument, ask for "train_or_predict"
    if len(sys.argv) == 1:
        train_or_predict = prompt_train_or_predict()

    # If user provided "train_or_predict" argument, assign the value to "train_or_predict"
    if len(sys.argv) >= 2:
        train_or_predict_input = sys.argv[1]
        train_or_predict = prompt_train_or_predict(train_or_predict_input)

    # If user did not provide "dataset_root_dir" argument, ask for "dataset_root_dir"
    if len(sys.argv) <= 2:
        dataset_root_dir = prompt_dataset_root_dir()

    # If user provided "dataset_root_dir" argument, assign the value to "dataset_root_dir"
    if len(sys.argv) >= 3:
        dataset_root_dir_input = sys.argv[2]
        dataset_root_dir = prompt_dataset_root_dir(dataset_root_dir_input)

    # If the did not provide "require_pre_process" argument, ask for "require_pre_process"
    if len(sys.argv) <= 3:
        require_pre_process = prompt_require_pre_process(dataset_root_dir)

    # If user provided "require_pre_process" argument, assign the value to "require_pre_process"
    if len(sys.argv) >= 4:
        require_pre_process_input = sys.argv[3]
        require_pre_process = prompt_require_pre_process(
            dataset_root_dir, require_pre_process_input
        )

    # If the user did not provide "weights_path" argument, ask for "weights_path"
    if len(sys.argv) <= 4:
        weights_path = prompt_weights_path(train_or_predict)

    # If user provided "weights_path" argument, assign the value to "weights_path"
    if len(sys.argv) >= 5:
        weights_path_input = sys.argv[4]
        weights_path = prompt_weights_path(train_or_predict, weights_path_input)

    # If the user did not provide all of the mandatory arguments, ask for optional "train_input_weights_path"
    if len(sys.argv) < 5:
        if train_or_predict == "Train":
            train_input_weights_path = prompt_train_input_weights_path()
        elif train_or_predict == "Predict":
            train_input_weights_path = "None"

    # If user provided "train_input_weights_path" argument, assign the value to "train_input_weights_path"
    if len(sys.argv) >= 6:
        train_input_weights_path = sys.argv[5]

        if (
            train_or_predict == "Predict"
            and train_input_weights_path
            not in valid_null_train_input_weights_path_responses
        ):
            print(
                f"Invalid value provided for 'train_input_weights_path': '{train_input_weights_path}'."
            )
            print(
                "This argument is for training, but the value of 'train_or_predict' chosen is 'Predict'."
            )
            print(
                f"Expected input when 'train_or_predict' is 'Predict': {valid_null_train_input_weights_path_responses}"
            )
            print("The provided value for 'train_input_weights_path' will be ignored.")
            print()
        elif (
            train_or_predict == "Train"
            and not train_input_weights_path == "None"
            and not (os.path.exists(train_input_weights_path))
        ):
            print(
                f"Invalid argument for 'train_input_weights_path': {train_input_weights_path}. Weights file not found."
            )
            print(
                f"Expected input: valid input weight file path or {valid_null_train_input_weights_path_responses}"
            )
            print()
            train_input_weights_path = prompt_train_input_weights_path()

    # If the user did not provide all of the mandatory arguments, ask for optional "seed"
    if len(sys.argv) < 5:
        seed = prompt_seed()

    # If user provided "seed" argument, assign the value to "seed"
    if len(sys.argv) == 7:
        if not (sys.argv[6].isnumeric()):
            print("Invalid argument for 'seed'. Expected input: integer.\n")
            seed = prompt_seed()
        else:
            seed = int(sys.argv[6])

    if len(sys.argv) <= 7:
        print("Selected parameters:")
        print("    train_or_predict:", train_or_predict)
        print("    dataset_root_dir:", dataset_root_dir)
        print("    require_pre_process:", require_pre_process)
        print("    weights_path:", weights_path)
        print("    train_input_weights_path:", train_input_weights_path)
        print("    seed:", seed)
        print()

        main(
            train_or_predict,
            dataset_root_dir,
            require_pre_process,
            weights_path,
            train_input_weights_path,  # Used as weights input for "train mode". Ignored for "predict" mode.
            seed,
        )
    else:
        print("Invalid usage.")
