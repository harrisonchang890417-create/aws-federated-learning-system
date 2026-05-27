"""Worker-side model utilities and FL loop."""

import io
import json
import logging
import os
import time
from pathlib import Path

import boto3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from collections import OrderedDict
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
SQS_WAIT_SECONDS = 20
MAX_WAIT_SECONDS = 180

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


def _build_dataloader(partition_id, labels_map, batch_size):
    partition_dir = Path(f"/home/ubuntu/fl-client/data_cache/client-{partition_id}")
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
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0), len(dataset)


def _parse_round_from_key(key):
    filename = os.path.basename(key)
    return int(filename.removeprefix("global_model_round_").removesuffix(".npz"))


def _wait_for_global_model(s3_client, sqs_client, queue_url, global_bucket, round_id):
    key = f"{MODELS_PREFIX}global_model_round_{round_id}.npz"
    deadline = time.time() + MAX_WAIT_SECONDS

    while time.time() < deadline:
        try:
            s3_client.head_object(Bucket=global_bucket, Key=key)
            return key
        except Exception:
            pass

        if not queue_url:
            time.sleep(2)
            continue

        resp = sqs_client.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=10,
            WaitTimeSeconds=SQS_WAIT_SECONDS,
            VisibilityTimeout=5,
        )
        for message in resp.get("Messages", []):
            matched = False
            try:
                payload = json.loads(message["Body"])
                if "Message" in payload:
                    payload = json.loads(payload["Message"])
                records = payload.get("Records", [])
                for record in records:
                    obj_key = record["s3"]["object"]["key"]
                    if obj_key == key or _parse_round_from_key(obj_key) >= round_id:
                        matched = True
            finally:
                sqs_client.delete_message(
                    QueueUrl=queue_url,
                    ReceiptHandle=message["ReceiptHandle"],
                )
            if matched:
                try:
                    s3_client.head_object(Bucket=global_bucket, Key=key)
                    return key
                except Exception:
                    continue

    raise TimeoutError(f"Timed out waiting for {key}")


def worker_main():
    torch.set_num_threads(1)

    asu_id = os.environ["ASU_ID"]
    partition_id = int(os.environ["PARTITION_ID"])
    worker_id = int(os.environ.get("WORKER_ID", partition_id))
    num_rounds = int(os.environ.get("NUM_ROUNDS", "5"))
    local_epochs = int(os.environ.get("LOCAL_EPOCHS", str(DEFAULT_EPOCHS)))
    batch_size = int(os.environ.get("BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))
    learning_rate = float(os.environ.get("LEARNING_RATE", str(DEFAULT_LR)))
    queue_url = os.environ.get("QUEUE_URL", "")

    session = boto3.Session(region_name=REGION)
    s3_client = session.client("s3")
    sqs_client = session.client("sqs")
    global_bucket = f"{asu_id}-global-bucket"
    local_bucket = f"{asu_id}-local-bucket"

    logger.info("Worker %s booting for partition %s", worker_id, partition_id)
    labels_map = _load_labels_map(s3_client, global_bucket)
    dataloader, dataset_size = _build_dataloader(partition_id, labels_map, batch_size)
    logger.info("Loaded %s samples for partition %s", dataset_size, partition_id)

    for round_id in range(num_rounds):
        global_key = _wait_for_global_model(
            s3_client,
            sqs_client,
            queue_url,
            global_bucket,
            round_id,
        )
        logger.info("Round %s starting from %s", round_id, global_key)
        resp = s3_client.get_object(Bucket=global_bucket, Key=global_key)
        global_state = deserialize_state_dict(resp["Body"].read())

        model = load_model(global_state)
        metrics = train_local(model, dataloader, learning_rate, local_epochs)
        update_key = f"{UPDATES_PREFIX}local_model_round_{round_id}_worker_{worker_id}.npz"
        s3_client.put_object(
            Bucket=local_bucket,
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

    logger.info("Worker %s finished all rounds", worker_id)


if __name__ == "__main__":
    worker_main()
