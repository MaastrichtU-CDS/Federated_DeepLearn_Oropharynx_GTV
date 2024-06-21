import json


class Params:
    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """"
        Save dict to json file

        Parameters
        ----------
        json_path : string
            Path to save location
        """
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """
        Load parameters from json file

        Parameters
        ----------
        json_path : string
            Path to json file
        """
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """"
        Give dict-like access to Params instance
        by: 'params.dict['learning_rate']'
        """
        return self.__dict__
