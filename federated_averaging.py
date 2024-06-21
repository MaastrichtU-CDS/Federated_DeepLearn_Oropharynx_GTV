from collections import OrderedDict
from typing import List, NoReturn
from collections import OrderedDict

import torch

from model import FastSmoothSENormDeepUNet_supervision_skip_no_drop as se_model


def calculate_federated_average(paths: List[str]) -> OrderedDict:
    """Calculate the federated average. Assumes at least 1 model.

    Parameters
    ----------
    paths : List[str]
        List of strings containings paths to models. These models will be
        averaged.

    Returns
    -------
    OrderedDict
        Average model stored as an OrderedDict.
    """

    # Load models and store them in a dict
    model_dict = {}
    for i, path in enumerate(paths):
        model_dict[str(i)] = torch.load(path)

    # Sum each model that was stored in 'model_dict'
    for i in range(len(model_dict)):
        if i == 0:
            federated_average = model_dict[str(i)]
        else:
            for key in federated_average:
                federated_average[key] += model_dict[str(i)][key]

    # Calculate average
    for key in federated_average:
        federated_average[key] /= len(model_dict)

    return federated_average


def save_average_model(avg_model: OrderedDict,
                       save_path: str,
                       device: str) -> NoReturn:
    """Save a model in specified path.

    Parameters
    ----------
    avg_model : OrderedDict
        Model in OrderedDict format.
    save_path : str
        Path to save location.
    device : str
        Run through 'cuda' or 'cpu'

    Returns
    -------
    NoReturn
        Return Nothing.
    """
    model = se_model(in_channels=1,
                     n_cls=2,
                     n_filters=16).to(device=device)
    model.load_state_dict(avg_model)
    torch.save(model.state_dict(), save_path)
