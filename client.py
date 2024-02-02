import sys
import requests
import torch
from classifier import classifier

def load_model(address):
    response = requests.get(address + "/model")
    net = classifier.Net()
    net.load_state_dict(
        {key: torch.tensor(value) for key, value in response.json().items()}
    )
    return net


class Client:
    def __init__(self, address, data_loader):
        self.address = address
        self.data_loader = data_loader
        self.model = load_model(self.address)

    def update_model(self):
        global_model = load_model(self.address)
        state = global_model.state_dict()

        _ = classifier.train_model(self.model, self.data_loader, 2)

        payload = {
            key: (value - state[key]).tolist() for key, value in self.model.state_dict().items()
        }

        # send update to server
        requests.post(self.address + "/model", json=payload)


if __name__ == "__main__":
    address = sys.argv[1]
    
    sample_size = 4
    if len(sys.argv) >= 3:
        sample_size = int(sys.argv[2])

    client = Client(address, classifier.random_dataloader(sample_size))
    client.update_model()
