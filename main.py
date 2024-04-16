import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from evaluate import *
from model import *
from pre_process_data import *
from predict import *
from process_data import *
from train import *

# ----------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------
# valid inputs for model_name argument
valid_model_name_inputs = ResNet_models + EfficientNet_models

# valid inputs for train_test_predict argument
valid_train_inputs = ["", "train", "Train"]
valid_test_inputs = ["test", "Test"]
valid_predict_inputs = ["predict", "Predict"]

# valid inputs for require_pre_process argument
valid_is_require_pre_process_inputs = [
    "",
    "y",
    "Y",
    "yes",
    "Yes",
    "t",
    "T",
    "true",
    "True",
]
valid_not_require_pre_process_inputs = [
    "n",
    "N",
    "no",
    "No",
    "f",
    "F",
    "false",
    "False",
]

# valid inputs for whether or not the user wants to use a custom seed value
valid_is_using_custom_seed_inputs = ["y", "Y", "yes", "Yes"]
valid_not_using_custom_seed_inputs = ["", "n", "N", "no", "No"]

# valid inputs for initial_weights_path argument when the user does not want to use custom intial weights
valid_null_initial_weights_path_inputs = [
    "none",
    "None",
    "null",
    "Null",
    "false",
    "False",
]

# valid inputs for whether or not the user wants to use initial weights for training
valid_is_using_initial_weights_inputs = ["y", "Y", "yes", "Yes"]
valid_is_not_using_initial_weights_inputs = ["", "n", "N", "no", "No"]


# ----------------------------------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------------------------------
def prompt_model_name(model_name="!invalid!") -> str:
    # initial check if model_name was provided
    if model_name != "!invalid!" and model_name not in valid_model_name_inputs:
        print(f"Invalid value provided for 'model_name': '{model_name}'.")
        print(
            f"Expected input: {valid_model_name_inputs}.",
        )
        print("Enter the correct value for the following input prompt...\n")

    # prompt if model_name is invalid
    while model_name not in valid_model_name_inputs:
        model_name = input("Provide the name of the model you wish to use: ")

        if model_name not in valid_model_name_inputs:
            print(f"Invalid model name. Expected input: {valid_model_name_inputs}.")

        print()

    return model_name


def prompt_train_test_predict(train_test_predict_input="!invalid!") -> str:
    # initial check if train_test_predict_input was provided
    if train_test_predict_input != "!invalid!" and train_test_predict_input not in (
        valid_train_inputs + valid_test_inputs + valid_predict_inputs
    ):
        print(
            f"Invalid value provided for 'train_test_predict': '{train_test_predict_input}'."
        )
        print(
            f"Expected input: {valid_train_inputs + valid_test_inputs + valid_predict_inputs}.",
        )
        print("Enter the correct value for the following input prompt...\n")

    # prompt if train_test_predict_input is invalid
    while train_test_predict_input not in (
        valid_train_inputs + valid_test_inputs + valid_predict_inputs
    ):
        train_test_predict_input = input(
            "Do you want to train the model, test the model, or make predictions (Train/Test/Predict)? "
        )

        if train_test_predict_input not in (
            valid_train_inputs + valid_test_inputs + valid_predict_inputs
        ):
            print(
                f"Invalid response. Expected input: {valid_train_inputs + valid_test_inputs + valid_predict_inputs}."
            )

        print()

    if train_test_predict_input == "":
        print("Default selection: Train")

    if train_test_predict_input in valid_train_inputs:
        train_test_predict = "Train"
    elif train_test_predict_input in valid_test_inputs:
        train_test_predict = "Test"
    elif train_test_predict_input in valid_predict_inputs:
        train_test_predict = "Predict"

    return train_test_predict


def prompt_dataset_root_dir(train_test_predict, dataset_root_dir="!invalid!") -> str:
    # initial check if dataset_root_dir was provided
    if dataset_root_dir != "!invalid!" and not (os.path.exists(dataset_root_dir)):
        print(
            f"Invalid value provided for 'dataset_root_dir': '{dataset_root_dir}'. Directory not found."
        )
        print("Enter the correct value for the following input prompt...\n")
    # if the user is not predicting, check for real and fake subdirectories
    elif (
        train_test_predict not in valid_predict_inputs
        and dataset_root_dir != "!invalid!"
        and (os.path.exists(dataset_root_dir))
    ):
        if not (os.path.exists(f"{dataset_root_dir}/real")):
            print(
                f"Invalid dataset root directory path. Subdirectory '{dataset_root_dir}/real' not found."
            )
        if not (os.path.exists(f"{dataset_root_dir}/fake")):
            print(
                f"Invalid dataset root directory path. Subdirectory '{dataset_root_dir}/fake' not found."
            )
        if not (os.path.exists(f"{dataset_root_dir}/real")) or not (
            (os.path.exists(f"{dataset_root_dir}/fake"))
        ):
            dataset_root_dir = "!invalid!"
            print()

    # prompt if dataset_root_dir is invalid
    while not (os.path.exists(dataset_root_dir)):
        dataset_root_dir = input(
            "Provide the relative path to the dataset root directory: "
        )

        if not (os.path.exists(dataset_root_dir)):
            print(
                f"Invalid dataset root directory path. Directory '{dataset_root_dir}' not found."
            )

        # if the user is not predicting, check for real and fake subdirectories
        elif train_test_predict not in valid_predict_inputs:
            if not (os.path.exists(f"{dataset_root_dir}/real")):
                print(
                    f"Invalid dataset root directory path. Subdirectory '{dataset_root_dir}/real' not found."
                )
            if not (os.path.exists(f"{dataset_root_dir}/fake")):
                print(
                    f"Invalid dataset root directory path. Subdirectory '{dataset_root_dir}/fake' not found."
                )
            if not (os.path.exists(f"{dataset_root_dir}/real")) or not (
                (os.path.exists(f"{dataset_root_dir}/fake"))
            ):
                dataset_root_dir = "!invalid!"

        print()

    return dataset_root_dir


def prompt_require_pre_process(
    dataset_root_dir: str, require_pre_process_input="!invalid!"
) -> bool:
    # initial check if require_pre_process_input was provided
    if (
        not (require_pre_process_input == "!invalid!")
        and require_pre_process_input not in valid_is_require_pre_process_inputs
        and require_pre_process_input not in valid_not_require_pre_process_inputs
    ):
        print(
            f"Invalid value provided for 'require_pre_process': '{require_pre_process_input}'."
        )
        print(
            f"Expected input: {valid_is_require_pre_process_inputs + valid_not_require_pre_process_inputs}."
        )
        print("Enter the correct value for the following input prompt...\n")

    if require_pre_process_input in valid_is_require_pre_process_inputs:
        require_pre_process = True
    elif require_pre_process_input in valid_not_require_pre_process_inputs:
        require_pre_process = False

    # prompt if require_pre_process_input is invalid
    while (
        require_pre_process_input not in valid_is_require_pre_process_inputs
        and require_pre_process_input not in valid_not_require_pre_process_inputs
    ):
        require_pre_process_input = input(
            "Do you need to perform pre-processing on the dataset (Y/N)? "
        )

        if (
            require_pre_process_input not in valid_is_require_pre_process_inputs
            and require_pre_process_input not in valid_not_require_pre_process_inputs
        ):
            print(
                f"Invalid response. Expected input: {valid_is_require_pre_process_inputs + valid_not_require_pre_process_inputs}."
            )
        else:
            if require_pre_process_input == "":
                print("Default selection: Yes")

            if require_pre_process_input in valid_is_require_pre_process_inputs:
                require_pre_process = True
            elif require_pre_process_input in valid_not_require_pre_process_inputs:
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


def prompt_weights_path(train_test_predict: bool, weights_path="!invalid!") -> str:
    # initial check if weights_path was provided
    weights_dir = "/".join(weights_path.split("/")[0:-1])

    if (
        train_test_predict == "Train"
        and weights_path != "!invalid!"
        and not (os.path.exists(weights_dir))
    ):
        print(
            f"Invalid value provided for 'weights_path': '{weights_path}'. Directory '{weights_dir}' not found."
        )
        print("Expected input: <weights_dir>/<weights_file>.")
        print("Enter the correct value for the following input prompt...\n")
    elif (
        (train_test_predict == "Test" or train_test_predict == "Predict")
        and weights_path != "!invalid!"
        and not (os.path.exists(weights_path))
    ):
        print(
            f"Invalid value provided for 'weights_path': '{weights_path}'. File not found."
        )
        print("Enter the correct value for the following input prompt...\n")

    if train_test_predict == "Train" and os.path.exists(weights_dir):
        try:
            with open(weights_path, "w"):
                pass
        except:
            print("Error: could not create the output weights file.\n")

        if not (os.path.exists(weights_path)):
            print("Error: could not create the output weights file.\n")

    # prompt if weights_path is invalid
    # If the user is training the model, ask for relative path to where the user wants to save the weights
    if train_test_predict == "Train":
        while not (os.path.exists(weights_path)):
            weights_path = input(
                "Provide the relative path to the output weights file: "
            )
            weights_dir = "/".join(weights_path.split("/")[0:-1])

            if not (os.path.exists(weights_dir)):
                print(
                    f"Invalid output weights file path. Directory '{weights_dir}' not found."
                )
                print("Expected input: <weights_dir>/<weights_file>.")
            else:
                try:
                    with open(weights_path, "w"):
                        pass
                except:
                    print("Error: could not create the output weights file.")

                if not (os.path.exists(weights_path)):
                    print("Error: could not create the output weights file.")

            print()

    # Else if the user is using the model to test or predict, ask for relative path to the pre-trained weights file
    elif train_test_predict == "Test" or train_test_predict == "Predict":
        while not (os.path.exists(weights_path)) or os.path.isdir(weights_path):
            weights_path = input(
                "Provide the relative path to the pre-trained weights file: "
            )

            if not (os.path.exists(weights_path)):
                print("Invalid pre-trained weights file path. File not found.")

            if os.path.isdir(weights_path):
                print(
                    "Invalid pre-trained weights file path. Expected file path but received directory path."
                )

            print()

    return weights_path


def prompt_initial_weights_path(initial_weights_path="!invalid!") -> str:
    # initial check if initial_weights_path was provided
    is_using_initial_weights_input = "!invalid!"
    is_using_initial_weights = False

    if (
        initial_weights_path != "!invalid!"  # if the user has provided a weights path
        and initial_weights_path
        not in valid_null_initial_weights_path_inputs  # and the weights path is not "None"
        and not (
            os.path.exists(initial_weights_path)
        )  # and the weights path does not exist
    ):
        print(
            f"Invalid argument for 'initial_weights_path': {initial_weights_path}. Weights file not found."
        )
        print(
            f"Expected input: valid initial weights file path or {valid_null_initial_weights_path_inputs}."
        )
        print("Enter the correct value for the following input prompt...\n")
        is_using_initial_weights_input = "Y"
        is_using_initial_weights = True
    elif os.path.isdir(initial_weights_path):
        print(
            f"Invalid argument for 'initial_weights_path': {initial_weights_path}. Expected file path but received directory path."
        )
        print("Enter the correct value for the following input prompt...\n")
    elif initial_weights_path in valid_null_initial_weights_path_inputs:
        is_using_initial_weights_input = "N"
        is_using_initial_weights = False

    # prompt if initial_weights_path was not provided
    while (
        is_using_initial_weights_input not in valid_is_using_initial_weights_inputs
        and is_using_initial_weights_input
        not in valid_is_not_using_initial_weights_inputs
    ):
        is_using_initial_weights_input = input(
            "Do you want to start training the model from previously saved weights (Y/N)? "
        )

        if is_using_initial_weights_input == "":
            print("Default selection: No")

        if is_using_initial_weights_input in valid_is_using_initial_weights_inputs:
            is_using_initial_weights = True
        elif (
            is_using_initial_weights_input in valid_is_not_using_initial_weights_inputs
        ):
            is_using_initial_weights = False
        else:
            print(
                f"Invalid response. Expected input: {valid_is_using_initial_weights_inputs + valid_is_not_using_initial_weights_inputs}.\n"
            )

    # prompt if initial_weights_path was provided but not valid
    if is_using_initial_weights:
        # Ask for relative path to the initial weights file
        while not (os.path.exists(initial_weights_path)) or (
            os.path.isdir(initial_weights_path)
        ):
            initial_weights_path = input(
                "Provide the relative path to your initial weights file: "
            )

            if not (os.path.exists(initial_weights_path)):
                print("Invalid initial weights file path. File not found.\n")

            if os.path.isdir(initial_weights_path):
                print(
                    "Invalid initial weights file path. Expected file path but received directory path.\n"
                )
    else:
        initial_weights_path = "None"

    print()

    return initial_weights_path


def prompt_seed(custom_seed_input="!invalid!") -> int:
    # initial check if custom_seed_input was provided
    is_using_custom_seed_input = "!invalid!"
    is_using_custom_seed = False
    custom_seed = 33

    if custom_seed_input != "!invalid!":
        is_using_custom_seed_input = "Y"
        is_using_custom_seed = True

        if not (custom_seed_input.isnumeric()):
            print(
                f"Invalid value provided for 'seed': '{custom_seed_input}'. Expected input: integer."
            )
            print("Enter the correct value for the following input prompt...\n")

    # prompt if custom_seed_input is invalid
    while (
        is_using_custom_seed_input not in valid_is_using_custom_seed_inputs
        and is_using_custom_seed_input not in valid_not_using_custom_seed_inputs
    ):
        is_using_custom_seed_input = input(
            "Do you wish to use a custom seed value (Y/N)? "
        )

        if is_using_custom_seed_input == "":
            print("Default selection: No")

        if is_using_custom_seed_input in valid_is_using_custom_seed_inputs:
            is_using_custom_seed = True
        elif is_using_custom_seed_input in valid_not_using_custom_seed_inputs:
            is_using_custom_seed = False
            print()
        else:
            print(
                f"Invalid response. Expected input: {valid_is_using_custom_seed_inputs + valid_not_using_custom_seed_inputs}.\n"
            )

    if is_using_custom_seed:
        while not (custom_seed_input.isnumeric()):
            custom_seed_input = input("Provide the seed (int): ")

            if not (custom_seed_input.isnumeric()):
                print("Invalid response. Expected input: integer.\n")
            else:
                custom_seed = int(custom_seed_input)
                print()

        custom_seed = int(custom_seed_input)
    else:
        custom_seed = 33

    return custom_seed


# ----------------------------------------------------------------------------------------------------
# Main Function
# ----------------------------------------------------------------------------------------------------
def main(
    model_name: str,
    train_test_predict: str,
    dataset_root_dir: str,
    require_pre_process: bool,
    weights_path: str,
    initial_weights_path: str = "None",
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
    print("Processor information:")
    print("    CUDA is available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("    Using CUDA device:", torch.cuda.get_device_name())
    else:
        device = torch.device("cpu")
        print("    Using CPU")

    print()

    # ----------------------------------------------------------------------------------------------------
    # Pre-process Data - extract frames and faces from the dataset root directory
    # ----------------------------------------------------------------------------------------------------
    if require_pre_process:
        extract_frames_and_faces(dataset_root_dir)

    # ----------------------------------------------------------------------------------------------------
    # Process Data - create data loaders from the pre-processed data
    # ----------------------------------------------------------------------------------------------------
    if train_test_predict in valid_train_inputs:
        train_loader, val_loader, test_loader = get_train_data_loaders(
            dataset_root_dir, device, seed
        )
    elif train_test_predict in valid_test_inputs:
        test_loader = get_test_data_loader(dataset_root_dir, device)

    # ----------------------------------------------------------------------------------------------------
    # Model - initialise the model
    # ----------------------------------------------------------------------------------------------------
    model = get_model(model_name, device)

    # ----------------------------------------------------------------------------------------------------
    # Train - train the model
    # ----------------------------------------------------------------------------------------------------
    # if training, train the model
    if train_test_predict in valid_train_inputs:
        # if the optional initial_weights_path is provided, load initial weights
        if (
            initial_weights_path not in valid_null_initial_weights_path_inputs
            and os.path.exists(initial_weights_path)
            and os.path.isfile(initial_weights_path)
        ):
            model.load_state_dict(torch.load(initial_weights_path))

        criterion = nn.CrossEntropyLoss()

        # Vary optimizer function
        # optimizer = optim.SGD(model.parameters(), lr=0.001)
        # optimizer = optim.Adadelta(model.parameters(), lr=0.001)
        # optimizer = optim.RMSprop(model.parameters(), lr=0.001)

        # Vary learning rate
        # optimizer = optim.Adam(model.parameters(), lr=0.005)
        # optimizer = optim.Adam(model.parameters(), lr=0.001)
        # optimizer = optim.Adam(model.parameters(), lr=0.0005)
        # optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # Vary L2 regularization via weight_decay
        # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)
        # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
        # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
        # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
        # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.000001)
        # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0000001)

        if model_name in ResNet_models:
            optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        if model_name in EfficientNet_models:
            optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

        model = train_model(model, criterion, optimizer, train_loader, 15, device)

        # Save weights
        torch.save(model.state_dict(), weights_path)
    # else if testing or predicting, load the weights
    elif train_test_predict in (valid_test_inputs + valid_predict_inputs):
        model.load_state_dict(torch.load(weights_path))

    # ----------------------------------------------------------------------------------------------------
    # Evaluate
    # ----------------------------------------------------------------------------------------------------
    # if using training mode, perform evaluation on validation set and show validation scores
    if train_test_predict in valid_train_inputs:
        val_accuracy, val_precision, val_recall, val_f1 = evaluate(
            model, val_loader, device
        )
        print("Validation Scores:")
        print(f"    Accuracy: {val_accuracy:.4f}")
        print(f"    Precision: {val_precision:.4f}")
        print(f"    Recall: {val_recall:.4f}")
        print(f"    F1 Score: {val_f1:.4f}")
        print()

    # if using training mode or testing mode, perform evaluation on test set show test scores
    if train_test_predict in (valid_train_inputs + valid_test_inputs):
        test_accuracy, test_precision, test_recall, test_f1 = evaluate(
            model, test_loader, device
        )
        print("Test Scores:")
        print(f"    Accuracy: {test_accuracy:.4f}")
        print(f"    Precision: {test_precision:.4f}")
        print(f"    Recall: {test_recall:.4f}")
        print(f"    F1 Score: {test_f1:.4f}")
        print()

    # if using prediction mode, perform prediction
    elif train_test_predict in valid_predict_inputs:
        _, result, _ = predict(dataset_root_dir, model, device)
        print("Result:", result)
        print()


# ----------------------------------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    train_test_predict = "train"
    dataset_root_dir = ""
    require_pre_process = False
    weights_path = ""
    initial_weights_path = "None"
    seed = 33

    print()

    # If user did not provide all the mandatory arguments, show help message
    if len(sys.argv) < 6:
        print("Insufficient arguments provided.\n")
        print(
            "Usage: main.py <train_test_predict> <dataset_root_dir> <require_pre_process> <weights_path>[ <initial_weights_path>][ <seed>]"
        )
        print(
            "- Angle brackets <> refer to arguments that must be replaced with the desired values."
        )
        print("- Square brackets [] refer to optional arguments.\n")

        print("Proceeding to show input prompts for the missing arguments...\n")

    # If user did not provide "model_name" argument, ask for "model_name"
    if (len(sys.argv)) == 1:
        model_name = prompt_model_name()

    # If user provided "model_name" argument, assign the value to "model_name"
    if len(sys.argv) >= 2:
        model_name = prompt_model_name(sys.argv[1])

    # If user did not provide "train_test_predict" argument, ask for "train_test_predict"
    if len(sys.argv) <= 2:
        train_test_predict = prompt_train_test_predict()

    # If user provided "train_test_predict" argument, assign the value to "train_test_predict"
    if len(sys.argv) >= 3:
        train_test_predict = prompt_train_test_predict(sys.argv[2])

    # If user did not provide "dataset_root_dir" argument, ask for "dataset_root_dir"
    if len(sys.argv) <= 3:
        dataset_root_dir = prompt_dataset_root_dir(train_test_predict)

    # If user provided "dataset_root_dir" argument, assign the value to "dataset_root_dir"
    if len(sys.argv) >= 4:
        dataset_root_dir = prompt_dataset_root_dir(train_test_predict, sys.argv[3])

    # If the did not provide "require_pre_process" argument, ask for "require_pre_process"
    if len(sys.argv) <= 4:
        require_pre_process = prompt_require_pre_process(dataset_root_dir)

    # If user provided "require_pre_process" argument, assign the value to "require_pre_process"
    if len(sys.argv) >= 5:
        require_pre_process = prompt_require_pre_process(dataset_root_dir, sys.argv[4])

    # If the user did not provide "weights_path" argument, ask for "weights_path"
    if len(sys.argv) <= 5:
        weights_path = prompt_weights_path(train_test_predict)

    # If user provided "weights_path" argument, assign the value to "weights_path"
    if len(sys.argv) >= 6:
        weights_path = prompt_weights_path(train_test_predict, sys.argv[5])

    # If the user did not provide all of the mandatory arguments, and if using the model in training mode,
    # ask for optional "initial_weights_path"
    if len(sys.argv) < 6:
        if train_test_predict == "Train":
            initial_weights_path = prompt_initial_weights_path()
        elif train_test_predict == "Test" or train_test_predict == "Predict":
            initial_weights_path = "None"

    # If user provided "initial_weights_path" argument, assign the value to "initial_weights_path"
    if len(sys.argv) >= 7:
        initial_weights_path = sys.argv[6]

        if (
            train_test_predict == "Test" or train_test_predict == "Predict"
        ) and initial_weights_path not in valid_null_initial_weights_path_inputs:
            print(
                f"Invalid value provided for 'initial_weights_path': '{initial_weights_path}'."
            )
            print(
                "This argument is for training, but the value of 'train_test_predict' chosen is 'Predict'."
            )
            print(
                f"Expected input when 'train_test_predict' is 'Predict': {valid_null_initial_weights_path_inputs}."
            )
            print("The provided value for 'initial_weights_path' will be ignored.\n")

            initial_weights_path = "None"
        elif train_test_predict == "Train":
            initial_weights_path = prompt_initial_weights_path(initial_weights_path)

    # If the user did not provide all of the mandatory arguments, ask for optional "seed"
    if len(sys.argv) < 6:
        if train_test_predict == "Train":
            seed = prompt_seed()

    # If user provided "seed" argument, assign the value to "seed"
    if len(sys.argv) == 8:
        seed = prompt_seed(sys.argv[7])

    if len(sys.argv) <= 8:
        print("Selected parameters:")
        print("    model_name:", model_name)
        print("    train_test_predict:", train_test_predict)
        print("    dataset_root_dir:", dataset_root_dir)
        print("    require_pre_process:", require_pre_process)
        print("    weights_path:", weights_path)
        print("    initial_weights_path:", initial_weights_path)
        print("    seed:", seed)
        print()

        main(
            model_name,
            train_test_predict,
            dataset_root_dir,
            require_pre_process,
            weights_path,
            initial_weights_path,
            seed,
        )
    else:
        print(
            f"Invalid usage: expected 5 to 7 arguments but received {len(sys.argv) - 1}.\n"
        )

# ----------------------------------------------------------------------------------------------------
# Training Cases
# ----------------------------------------------------------------------------------------------------
# Best ResNet architecture and EfficientNet architecture
# main(
#     "ResNet50",
#     "Train",
#     "dataset/train",
#     False,
#     "weights/ResNet50_Adam_LR1e-4_WD0_DP0_EP15.pth",
# )
# main(
#     "EfficientNetB2",
#     "Train",
#     "dataset/train",
#     False,
#     "weights/EfficientNetB2_Adam_LR1e-4_WD0_DP0_EP15.pth",
# )

# ResNet Architecture Exploration
# main(
#     "ResNet18",
#     "Train",
#     "dataset/train",
#     False,
#     "weights/ResNet18_Adam_LR1e-3_EP15.pth",
# )
# main(
#     "ResNet34",
#     "Train",
#     "dataset/train",
#     False,
#     "weights/ResNet34_Adam_LR1e-3_EP15.pth",
# )
# main(
#     "ResNet50",
#     "Train",
#     "dataset/train",
#     False,
#     "weights/ResNet50_Adam_LR1e-3_EP15.pth",
# )
# main(
#     "ResNet101",
#     "Train",
#     "dataset/train",
#     False,
#     "weights/ResNet101_Adam_LR1e-3_EP15.pth",
# )
# main(
#     "ResNet152",
#     "Train",
#     "dataset/train",
#     False,
#     "weights/ResNet152_Adam_LR1e-3_EP15.pth",
# )

# EfficientNet Architecture Exploration
# main(
#     "EfficientNetB0",
#     "Train",
#     "dataset/train",
#     False,
#     "weights/EfficientNetB0_Adam_LR1e-3_EP15.pth",
# )
# main(
#     "EfficientNetB1",
#     "Train",
#     "dataset/train",
#     False,
#     "weights/EfficientNetB1_Adam_LR1e-3_EP15.pth",
# )
# main(
#     "EfficientNetB2",
#     "Train",
#     "dataset/train",
#     False,
#     "weights/EfficientNetB2_Adam_LR1e-3_EP15.pth",
# )
# main(
#     "EfficientNetB3",
#     "Train",
#     "dataset/train",
#     False,
#     "weights/EfficientNetB3_Adam_LR1e-3_EP15.pth",
# )
# main(
#     "EfficientNetB4",
#     "Train",
#     "dataset/train",
#     False,
#     "weights/EfficientNetB4_Adam_LR1e-3_EP15.pth",
# )
# main(
#     "EfficientNetB5",
#     "Train",
#     "dataset/train",
#     False,
#     "weights/EfficientNetB5_Adam_LR1e-3_EP15.pth",
# )


# ----------------------------------------------------------------------------------------------------
# Command Line Test Cases
# ----------------------------------------------------------------------------------------------------

# invalid model_name
# py main.py f

# valid model_name
# py main.py ResNet50

# invalid train_test_predict
# py main.py ResNet50 tra

# valid train_or predict
# py main.py ResNet50 train

# invalid dataset_root_dir
# py main.py ResNet50 train da

# invalid dataset_root_dir 2
# py main.py ResNet50 train dataset

# valid dataset_root_dir
# py main.py ResNet50 train dataset/train_small

# invalid require_pre_process
# py main.py ResNet50 train dataset/train_small asdf

# valid require_pre_process
# py main.py ResNet50 train dataset/train_small true

# invalid weights_path
# py main.py ResNet50 train dataset/train_small true abc

# valid weights_path - all mandatory args valid
# py main.py ResNet50 train dataset/train_small false weights/test_small.pth

# invalid initial_weights_path
# py main.py ResNet50 train dataset/train_small true weights/BCE_LR1e-3_EPOCH10.pth test

# valid initial_weights_path
# py main.py ResNet50 train dataset/train_small true weights/BCE_LR1e-3_EPOCH10.pth None

# valid initial_weights_path 2
# py main.py ResNet50 predict dataset/train_small true weights/test_small.pth weights/BCE_LR1e-3_EPOCH10.pth

# invalid seed
# py main.py ResNet50 predict dataset/train_small true weights/test_small.pth none a

# valid seed, all mandatory and optional args valid
# py main.py ResNet50 predict dataset/train_small true weights/test_small.pth none 27

# too many args
# py main.py ResNet50 predict dataset/train_small true weights/test_small.pth none 27 tooManyArgs
