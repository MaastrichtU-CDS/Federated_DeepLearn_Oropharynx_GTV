# %%
import datetime
import os
import pathlib
from pathlib import Path
from typing import List, NoReturn, Tuple

from federated_averaging import calculate_federated_average, save_average_model
from train_3d_segmentation import train_main

from param_loader import Params


def absolute_file_paths(directory: str) -> List[str]:
    """Get absolute file paths of every item in directory.

    Parameters
    ----------
    directory : str
        Path

    Returns
    -------
    paths : List[str]
        List of strings containing absolute paths of every file in directory
    """

    paths = []
    for dirpath, _, filenames in os.walk(directory):
        for file in filenames:
            if file.endswith('highest_val.pt'):
                paths.append(os.path.abspath(os.path.join(dirpath, file)))
    return paths


def create_folder_structure(current_time: str,
                            params: dict) -> Tuple(Path, Path, Path):
    """Create folder structure for aggregated models, average model and
    log directories.

    Parameters
    ----------
    current_time : str
        Current time for labeling experiment.
    params : dict
        Dictionary containing base path locations.

    Returns
    -------
    save_model_path : Path
        Path to save location for aggregrate models.
    average_model_path : Path
        Path to save location for average models.
    save_logs_path : Path
        Path to save location for training and validation logs.
    """
    log_path = pathlib.Path(params.dict['log_path'])
    experiment_path = log_path / str(current_time)
    save_model_path = experiment_path / 'aggregate_models'
    average_model_path = experiment_path / 'average_models'
    save_logs_path = experiment_path / 'logs'
    save_model_path.mkdir(parents=True, exist_ok=False)
    average_model_path.mkdir(parents=True, exist_ok=False)
    save_logs_path.mkdir(parents=True, exist_ok=False)
    return save_model_path, average_model_path, save_logs_path


def run(params: dict) -> NoReturn:
    """Executes training and validation schema in a federated setting.

    Parameters
    ----------
    params : dict
        Dict containing all parameters and hyperparameters for training.

    Returns
    -------
    NoReturn

    """
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_model_path, average_model_path, save_logs_path = create_folder_structure(current_time,
                                                                                  params)

    for training_cycle in range(0, params.dict['number_of_federated_cycles']):
        print(f'Current federated cycle: {training_cycle}')
        # Starting Weights
        save_model_path_iteration = save_model_path / str(training_cycle)
        average_model_path_iteration = average_model_path / str(training_cycle)
        save_logs_path_iteration = save_logs_path / str(training_cycle)
        save_model_path_iteration.mkdir(parents=True, exist_ok=False)
        average_model_path_iteration.mkdir(parents=True, exist_ok=False)
        save_logs_path_iteration.mkdir(parents=True, exist_ok=False)

        for centre in params.dict['centres']:
            # Train
            print(f'Training on centre: {centre}')
            centre_data_path = os.path.join(params.dict['data_path'], centre)

            # Start from an average model if training cycle > 0
            if training_cycle != 0:
                load_path = average_model_path / str(training_cycle - 1)

                model_path = os.path.join(load_path,
                                          'average_model.pt')
            # In the first iteration model training will start from scratch.
            else:
                model_path = None

            train_main(centre_data_path,
                       model_path,
                       training_cycle,
                       save_model_path_iteration,
                       save_logs_path_iteration)

        # Federated Averaging
        print(f'Starting averaging...')
        paths = absolute_file_paths(save_model_path_iteration)
        print(paths)
        federated_model = calculate_federated_average(paths=paths)
        save_path = average_model_path_iteration / ('average_model.pt')
        save_average_model(avg_model=federated_model,
                           save_path=save_path,
                           device='cuda')


if __name__ == "__main__":
    param_path = pathlib.Path(os.getcwd() + '/federated_params.json')
    params = Params(param_path)
    run(params=params)

# %%
