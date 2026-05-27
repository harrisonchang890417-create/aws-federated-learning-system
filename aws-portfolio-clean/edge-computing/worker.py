"""Greengrass MQTT-driven federated learning worker."""

import io
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

import boto3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from awsiot.greengrasscoreipc.clientv2 import GreengrassCoreIPCClientV2
from awsiot.greengrasscoreipc.model import QOS
from torch.utils.data import DataLoader, TensorDataset


NUM_CLASSES = 10
REGION = "us-west-2"
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
MODELS_PREFIX = "models/"
UPDATES_PREFIX = "updates/"
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 0.01
DEFAULT_EPOCHS = 5
DEFAULT_NUM_ROUNDS = 5
POLL_SECONDS = 2
MAX_WAIT_SECONDS = 300
STATE_DIR = Path("/tmp")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("worker")


class LeNet5(nn.Module):
    """LeNet-5 for MNIST."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_model(num_classes=NUM_CLASSES):
    return LeNet5(num_classes=num_classes)


def load_model(state_dict, num_classes=NUM_CLASSES):
    model = LeNet5(num_classes=num_classes)
    model.load_state_dict(state_dict)
    return model


def serialize_state_dict(state_dict):
    buf = io.BytesIO()
    np.savez(buf, **{k: v.cpu().numpy() for k, v in state_dict.items()})
    return buf.getvalue()


def deserialize_state_dict(data):
    npz = np.load(io.BytesIO(data))
    return OrderedDict({k: torch.from_numpy(npz[k]) for k in npz.files})


def train_local(model, dataloader, lr, epochs):
    device = torch.device("cpu")
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for _ in range(epochs):
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    return {
        "train_loss": total_loss / max(total_samples, 1),
        "train_accuracy": total_correct / max(total_samples, 1),
        "num_samples": total_samples // max(epochs, 1),
    }


def _load_labels_map(s3_client, global_bucket):
    resp = s3_client.get_object(Bucket=global_bucket, Key="labels.csv")
    content = resp["Body"].read().decode()
    labels_map = {}
    for line in content.strip().splitlines()[1:]:
        parts = line.strip().split(",")
        labels_map[parts[0]] = int(parts[2])
    return labels_map


def _transform_image(path):
    img = Image.open(path).convert("L").resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MNIST_MEAN) / MNIST_STD
    return arr.reshape(1, 28, 28)


def _load_state(state_path):
    if not state_path.exists():
        return {"completed_rounds": []}
    try:
        return json.loads(state_path.read_text())
    except Exception:
        return {"completed_rounds": []}


def _save_state(state_path, state):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, sort_keys=True))


def _wait_for_global_model(s3_client, global_bucket, round_id):
    key = f"{MODELS_PREFIX}global_model_round_{round_id}.npz"
    deadline = time.time() + MAX_WAIT_SECONDS
    while time.time() < deadline:
        try:
            s3_client.head_object(Bucket=global_bucket, Key=key)
            return key
        except Exception:
            time.sleep(POLL_SECONDS)
    raise TimeoutError(f"Timed out waiting for {key}")


def _round_from_payload(payload_bytes, completed_rounds):
    if not payload_bytes:
        return max(completed_rounds, default=-1) + 1

    text = payload_bytes.decode("utf-8", errors="ignore").strip()
    if not text:
        return max(completed_rounds, default=-1) + 1

    if text.isdigit():
        return int(text)

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return max(completed_rounds, default=-1) + 1

    for key in ("round", "round_id", "next_round", "round_number"):
        if key in payload:
            return int(payload[key])

    return max(completed_rounds, default=-1) + 1


class GreengrassWorker:
    """Persistent MQTT-driven wrapper for the FL worker logic."""

    def __init__(self):
        torch.set_num_threads(1)

        self.asu_id = os.environ["ASU_ID"]
        self.partition_id = int(os.environ["PARTITION_ID"])
        self.worker_id = int(os.environ.get("WORKER_ID", self.partition_id))
        self.num_rounds = int(os.environ.get("NUM_ROUNDS", str(DEFAULT_NUM_ROUNDS)))
        self.local_epochs = int(os.environ.get("LOCAL_EPOCHS", str(DEFAULT_EPOCHS)))
        self.batch_size = int(os.environ.get("BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))
        self.learning_rate = float(os.environ.get("LEARNING_RATE", str(DEFAULT_LR)))
        self.topic = os.environ.get("MQTT_TOPIC", f"fl/{self.asu_id}/next-round")
        self.data_root = os.environ.get("DATA_ROOT", "/home/ubuntu/fl-client/data_cache")
        self.round_coordinator = int(os.environ.get("ROUND_COORDINATOR_ID", "0"))

        self.global_bucket = f"{self.asu_id}-global-bucket"
        self.local_bucket = f"{self.asu_id}-local-bucket"
        self.session = None
        self.s3_client = None
        self.dataloader = None
        self.dataset_size = None

        self.state_path = STATE_DIR / f"com.fl.Worker-worker-{self.worker_id}.json"
        self.state_lock = threading.Lock()
        self.inflight_rounds = set()
        self.ipc_client = None

    def _build_dataloader(self, labels_map):
        partition_dir = Path(self.data_root) / f"client-{self.partition_id}"
        image_paths = sorted(partition_dir.glob("*.png"))
        if not image_paths:
            raise RuntimeError(f"No training images found in {partition_dir}")

        images = []
        labels = []
        for image_path in image_paths:
            if image_path.name not in labels_map:
                raise RuntimeError(f"Missing label for {image_path.name}")
            images.append(_transform_image(image_path))
            labels.append(labels_map[image_path.name])

        x_tensor = torch.tensor(np.stack(images), dtype=torch.float32)
        y_tensor = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(x_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        return dataloader, len(dataset)

    def _mark_round_completed(self, round_id):
        with self.state_lock:
            state = _load_state(self.state_path)
            completed = set(state.get("completed_rounds", []))
            completed.add(round_id)
            state["completed_rounds"] = sorted(completed)
            _save_state(self.state_path, state)
            self.inflight_rounds.discard(round_id)

    def _ensure_runtime_ready(self):
        if self.s3_client is None:
            self.session = boto3.Session(region_name=REGION)
            self.s3_client = self.session.client("s3")

        if self.dataloader is None:
            labels_map = _load_labels_map(self.s3_client, self.global_bucket)
            self.dataloader, self.dataset_size = self._build_dataloader(labels_map)
            logger.info(
                "Loaded %s samples for partition %s",
                self.dataset_size,
                self.partition_id,
            )

    def _should_process_round(self, round_id):
        with self.state_lock:
            state = _load_state(self.state_path)
            completed = set(state.get("completed_rounds", []))
            if round_id in completed or round_id in self.inflight_rounds:
                return False
            self.inflight_rounds.add(round_id)
            return True

    def _publish_next_round(self, round_id):
        payload = json.dumps({"round": round_id}).encode("utf-8")
        self.ipc_client.publish_to_iot_core(
            topic_name=self.topic,
            qos=QOS.AT_LEAST_ONCE,
            payload=payload,
        )
        logger.info("Published trigger for round %s", round_id)

    def _run_round(self, round_id):
        if round_id >= self.num_rounds:
            logger.info("Ignoring round %s because num_rounds=%s", round_id, self.num_rounds)
            return
        if not self._should_process_round(round_id):
            logger.info("Skipping duplicate/inflight round %s", round_id)
            return

        try:
            self._ensure_runtime_ready()
            global_key = _wait_for_global_model(self.s3_client, self.global_bucket, round_id)
            logger.info("Round %s starting from %s", round_id, global_key)
            resp = self.s3_client.get_object(Bucket=self.global_bucket, Key=global_key)
            global_state = deserialize_state_dict(resp["Body"].read())

            model = load_model(global_state)
            metrics = train_local(model, self.dataloader, self.learning_rate, self.local_epochs)
            update_key = f"{UPDATES_PREFIX}local_model_round_{round_id}_worker_{self.worker_id}.npz"
            self.s3_client.put_object(
                Bucket=self.local_bucket,
                Key=update_key,
                Body=serialize_state_dict(model.state_dict()),
            )
            logger.info(
                "Uploaded %s with loss=%.4f acc=%.4f samples=%s",
                update_key,
                metrics["train_loss"],
                metrics["train_accuracy"],
                metrics["num_samples"],
            )
            self._mark_round_completed(round_id)

            if round_id + 1 < self.num_rounds:
                self._publish_next_round(round_id + 1)
        except Exception:
            with self.state_lock:
                self.inflight_rounds.discard(round_id)
            logger.exception("Round %s failed", round_id)

    def _handle_stream_event(self, event):
        message = event.message
        payload = getattr(message, "payload", b"")
        completed_rounds = _load_state(self.state_path).get("completed_rounds", [])
        round_id = _round_from_payload(payload, completed_rounds)
        logger.info("Received MQTT trigger payload=%r resolved_round=%s", payload, round_id)
        threading.Thread(target=self._run_round, args=(round_id,), daemon=True).start()

    @staticmethod
    def _handle_stream_error(error):
        logger.error("MQTT stream error: %s", error)
        return True

    @staticmethod
    def _handle_stream_closed():
        logger.warning("MQTT stream closed")

    def serve_forever(self):
        while True:
            try:
                self.ipc_client = GreengrassCoreIPCClientV2()
                logger.info(
                    "Worker %s subscribing to %s for partition %s",
                    self.worker_id,
                    self.topic,
                    self.partition_id,
                )
                self.ipc_client.subscribe_to_iot_core(
                    topic_name=self.topic,
                    qos=QOS.AT_LEAST_ONCE,
                    on_stream_event=self._handle_stream_event,
                    on_stream_error=self._handle_stream_error,
                    on_stream_closed=self._handle_stream_closed,
                )
                while True:
                    time.sleep(60)
            except Exception:
                logger.exception("Worker service loop failed; retrying in 10 seconds")
                time.sleep(10)


def worker_main():
    GreengrassWorker().serve_forever()


if __name__ == "__main__":
    worker_main()
