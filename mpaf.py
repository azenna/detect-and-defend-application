import requests

from classifier import classifier
from client import load_model

class MPAF:
    def __init__(self, address, scaling):

        self.address = address
        self.scaling = scaling

        self.target_state = classifier.Net().state_dict()

    def send_bad_update(self):

        global_model = load_model(self.address)
        global_state = global_model.state_dict()

        payload = {key: ((self.target_state[key] - value) * self.scaling).tolist() for key, value in global_state.items()}
        requests.post(self.address + "/model", json=payload)

