import sys
import requests
from classifier import classifier
import client
import mpaf
import multiprocessing
import random

class ClientSim:
    def __init__(self, address, percent_evil, num_clients, scaling, train_sample_size):
        self.address = address
        self.percent_evil = percent_evil

        self.good_clients = self.make_good_clients(
            address, round(num_clients * (1 - self.percent_evil)), 4
        )

        self.mpaf = mpaf.MPAF(address, scaling)

    def send_updates(self):

        for i, c in enumerate(self.good_clients):
            c.update_model()
            if i % round((1 - self.percent_evil) / self.percent_evil) == 0:
                self.mpaf.send_bad_update()
        

    def make_good_clients(self, address, num, training_size):
        return [
            client.Client(address, classifier.random_dataloader(training_size))
            for i in range(num)
        ]

    def report(self):
        historical_accuracy = requests.get(address + "/model/accuracy")
        print(historical_accuracy.json())


if __name__ == "__main__":

    num_args = len(sys.argv)
    address = sys.argv[1]

    percent_evil = .2
    num_clients = 100
    evil_scale = 1
    train_sample_size = 4

    if num_args >= 3:
        percent_evil = float(sys.argv[2])
    if num_args >= 4:
        num_clients = int(sys.argv[3])
    if num_args >= 5:
        evil_scale = float(sys.argv[4])
    if num_args >= 6:
        train_sample_size = float(sys.argv[5])

    client_sim = ClientSim(address, percent_evil, num_clients, evil_scale, train_sample_size)

    client_sim.send_updates()
    client_sim.report()
