"""Lambda-side aggregation and evaluation helpers."""

import io
import os
import json
import tarfile
import logging

import boto3
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aggregator")

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

MODELS_PREFIX = "models/"
UPDATES_PREFIX = "updates/"
METRICS_PREFIX = "metrics/"

s3_client = boto3.client("s3", region_name="us-west-2")
_cached_test_data = None

def federated_average(client_updates):
    if not client_updates:
        raise ValueError("No client updates to aggregate")

    total = sum(n for _, n in client_updates)
    if total == 0:
        raise ValueError("Total samples across all clients is 0")

    first = client_updates[0][0]
    result = {k: np.zeros_like(first[k], dtype=np.float64) for k in first}

    for sd, n in client_updates:
        w = n / total
        for k in result:
            result[k] += w * sd[k].astype(np.float64)

    return {k: v.astype(first[k].dtype) for k, v in result.items()}

def save_npz(state_dict):
    buf = io.BytesIO()
    np.savez(buf, **state_dict)
    return buf.getvalue()


def load_npz(data):
    npz = np.load(io.BytesIO(data))
    return {k: npz[k] for k in npz.files}

def _conv2d(x, w, b, pad=0):
    if pad > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    N, C, H, W = x.shape
    F, _, kH, kW = w.shape
    oH, oW = H - kH + 1, W - kW + 1
    out = np.zeros((N, F, oH, oW))
    for f in range(F):
        for i in range(oH):
            for j in range(oW):
                out[:, f, i, j] = np.sum(
                    x[:, :, i:i+kH, j:j+kW] * w[f], axis=(1, 2, 3)
                ) + b[f]
    return out


def _relu(x):
    return np.maximum(0, x)


def _max_pool2d(x, size=2):
    N, C, H, W = x.shape
    oH, oW = H // size, W // size
    out = np.zeros((N, C, oH, oW))
    for i in range(oH):
        for j in range(oW):
            out[:, :, i, j] = x[:, :,
                                i*size:(i+1)*size,
                                j*size:(j+1)*size].max(axis=(2, 3))
    return out


def _linear(x, w, b):
    return x @ w.T + b


def lenet5_forward(sd, images):
    x = images
    x = _max_pool2d(_relu(_conv2d(x, sd['conv1.weight'], sd['conv1.bias'], pad=2)), 2)
    x = _max_pool2d(_relu(_conv2d(x, sd['conv2.weight'], sd['conv2.bias'])), 2)
    x = x.reshape(x.shape[0], -1)
    x = _relu(_linear(x, sd['fc1.weight'], sd['fc1.bias']))
    x = _relu(_linear(x, sd['fc2.weight'], sd['fc2.bias']))
    x = _linear(x, sd['fc3.weight'], sd['fc3.bias'])
    return x

def cross_entropy_loss(logits, labels):
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(len(labels)), labels].mean())


def transform_image(img):
    img = img.convert("L").resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0
    return ((arr - MNIST_MEAN) / MNIST_STD).reshape(1, 28, 28)


def load_test_data(global_bucket):
    global _cached_test_data
    if _cached_test_data is not None:
        return _cached_test_data

    logger.info("Loading test set from S3...")

    # Labels
    resp = s3_client.get_object(Bucket=global_bucket, Key="labels.csv")
    content = resp["Body"].read().decode()
    labels_map = {}
    for line in content.strip().split("\n")[1:]:
        parts = line.strip().split(",")
        labels_map[parts[0]] = int(parts[2])

    # Test images
    resp = s3_client.get_object(Bucket=global_bucket, Key="archives/test.tar.gz")
    tar_bytes = resp["Body"].read()

    images = []
    targets = []
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".png"):
                continue
            filename = os.path.basename(member.name)
            if filename not in labels_map:
                continue
            f = tar.extractfile(member)
            img = Image.open(io.BytesIO(f.read()))
            images.append(transform_image(img))
            targets.append(labels_map[filename])

    images_np = np.concatenate(images, axis=0).reshape(len(images), 1, 28, 28)
    labels_np = np.array(targets, dtype=np.int64)
    _cached_test_data = (images_np, labels_np)
    logger.info(f"Test set cached: {len(images)} images")
    return _cached_test_data


def evaluate_model(sd, test_images, test_labels):
    logits = lenet5_forward(sd, test_images)
    preds = logits.argmax(axis=1)
    acc = float((preds == test_labels).mean())
    loss = cross_entropy_loss(logits, test_labels)
    return {
        "accuracy": acc,
        "loss": loss,
        "total": len(test_labels),
        "correct": int((preds == test_labels).sum()),
    }

def handler(event, context):
    logger.info("Received event: %s", json.dumps(event))

    records = event.get("Records", [])
    if not records:
        return {"statusCode": 200, "body": "No records to process"}

    asu_id = os.environ["ASU_ID"]
    num_clients = int(os.environ.get("NUM_CLIENTS", "10"))
    total_rounds = int(os.environ.get("TOTAL_ROUNDS", "5"))
    global_bucket = f"{asu_id}-global-bucket"
    local_bucket = f"{asu_id}-local-bucket"

    key = records[0]["s3"]["object"]["key"]
    filename = os.path.basename(key)
    if not filename.startswith("local_model_round_"):
        return {"statusCode": 200, "body": f"Ignored key: {key}"}

    round_id = int(filename.split("_")[3])
    if round_id >= total_rounds:
        return {"statusCode": 200, "body": f"Ignored out-of-range round: {round_id}"}

    aggregated_key = f"{MODELS_PREFIX}global_model_round_{round_id + 1}.npz"
    try:
        s3_client.head_object(Bucket=global_bucket, Key=aggregated_key)
        logger.info("Round %s already aggregated", round_id)
        return {"statusCode": 200, "body": f"Round {round_id} already aggregated"}
    except Exception:
        pass

    resp = s3_client.list_objects_v2(
        Bucket=local_bucket,
        Prefix=f"{UPDATES_PREFIX}local_model_round_{round_id}_worker_",
    )
    contents = sorted(resp.get("Contents", []), key=lambda item: item["Key"])
    if len(contents) < num_clients:
        logger.info(
            "Round %s incomplete: found %s/%s updates",
            round_id,
            len(contents),
            num_clients,
        )
        return {
            "statusCode": 200,
            "body": f"Waiting for more updates for round {round_id}",
        }

    client_updates = []
    for item in contents[:num_clients]:
        body = s3_client.get_object(Bucket=local_bucket, Key=item["Key"])["Body"].read()
        client_updates.append((load_npz(body), 1))

    global_sd = federated_average(client_updates)
    s3_client.put_object(
        Bucket=global_bucket,
        Key=aggregated_key,
        Body=save_npz(global_sd),
    )

    test_images, test_labels = load_test_data(global_bucket)
    eval_result = evaluate_model(global_sd, test_images, test_labels)
    metrics = {
        "round": round_id,
        "accuracy": eval_result["accuracy"],
        "loss": eval_result["loss"],
    }
    s3_client.put_object(
        Bucket=global_bucket,
        Key=f"{METRICS_PREFIX}round_{round_id}.json",
        Body=json.dumps(metrics).encode(),
        ContentType="application/json",
    )

    logger.info("Round %s complete: %s", round_id, metrics)
    return {
        "statusCode": 200,
        "body": json.dumps(metrics),
    }
