from flask import Flask
from flask import request
import torch
import sys
from classifier import classifier

class FederatedLearning:
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold
        self.client_updates = []
        self.historical_accuracy = [classifier.test_model(self.model)]

    def aggregate(self):
        state = self.model.state_dict()

        for k in state.keys():
            state[k] += torch.stack(
                [
                    self.client_updates[i][k].float()
                    for i in range(len(self.client_updates))
                ],
                0,
            ).mean(0)

        self.model.load_state_dict(state)


    def update(self, client_update):
        self.client_updates.append(client_update)
        if len(self.client_updates) >= self.threshold:
            print("Aggregating updates:")

            self.aggregate()
            self.client_updates = []

            accuracy = classifier.test_model(self.model)
            self.historical_accuracy.append(accuracy)

            print(f"Model accuracy: {accuracy}")


def create_app(federated_model):
    app = Flask(__name__)

    @app.get("/model")
    def model_get():
        return {
            key: value.tolist()
            for key, value in federated_model.model.state_dict().items()
        }

    @app.post("/model")
    def model_post():
        federated_model.update(
            {key: torch.tensor(value) for key, value in request.get_json().items()}
        )
        return ("", 204)

    @app.get("/model/accuracy")
    def model_accuracy_get():
        return federated_model.historical_accuracy

    return app

if __name__ == "__main__":
    net = classifier.Net()
    net.load_state_dict(torch.load("./classifier/classifier.pth"))

    threshold = 100
    if len(sys.argv) >= 2:
        threshold = int(sys.argv[1])

    federated_model = FederatedLearning(net, threshold)
    app = create_app(federated_model)

    app.run()
