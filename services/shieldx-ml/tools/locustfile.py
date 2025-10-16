from locust import HttpUser, task, between
import json
import numpy as np


class InferenceUser(HttpUser):
    wait_time = between(0.001, 0.01)

    def on_start(self):
        # Prepare a small random payload (adjust input_dim as per model)
        self.input_dim = 50
        self.batch = 32
        self.model_name = "autoencoder_demo"

    @task(3)
    def predict(self):
        data = np.random.randn(self.batch, self.input_dim).tolist()
        payload = {"data": data, "return_proba": False}
        self.client.post(f"/models/{self.model_name}/predict", data=json.dumps(payload), headers={"Content-Type": "application/json"})

    @task(1)
    def health(self):
        self.client.get("/health")
