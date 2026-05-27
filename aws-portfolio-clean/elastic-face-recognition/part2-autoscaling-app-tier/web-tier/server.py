import os
import time
import uuid
import queue
import threading

import boto3
from flask import Flask, request

REGION = "us-west-2"
ASUID = "1236295946"
S3_BUCKET = f"{ASUID}-in-bucket"
REQ_QUEUE_URL = f"https://sqs.{REGION}.amazonaws.com/671194378296/{ASUID}-req-queue"
RESP_QUEUE_URL = f"https://sqs.{REGION}.amazonaws.com/671194378296/{ASUID}-resp-queue"
APP_TIER_NAME_PREFIX = "app-tier-instance"
MAX_APP_INSTANCES = 15

app = Flask(__name__)

s3 = boto3.client("s3", region_name=REGION)
sqs = boto3.client("sqs", region_name=REGION)
ec2 = boto3.client("ec2", region_name=REGION)
dynamodb = boto3.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(f"{ASUID}-dynamoDB")

response_waiters = {}
waiters_lock = threading.Lock()
scale_lock = threading.Lock()
last_scale_attempt = 0.0


def ensure_app_capacity():
    global last_scale_attempt

    now = time.time()
    with scale_lock:
        if now - last_scale_attempt < 2:
            return
        last_scale_attempt = now

    try:
        reservations = ec2.describe_instances(
            Filters=[
                {"Name": "tag:Name", "Values": [f"{APP_TIER_NAME_PREFIX}*"]},
                {
                    "Name": "instance-state-name",
                    "Values": ["pending", "running", "stopped"],
                },
            ]
        )["Reservations"]

        instances = [
            instance
            for reservation in reservations
            for instance in reservation["Instances"]
        ]
        live_ids = [
            instance["InstanceId"]
            for instance in instances
            if instance["State"]["Name"] in {"pending", "running"}
        ]
        if len(live_ids) >= MAX_APP_INSTANCES:
            return

        stopped_ids = [
            instance["InstanceId"]
            for instance in instances
            if instance["State"]["Name"] == "stopped"
        ]
        to_start = stopped_ids[: MAX_APP_INSTANCES - len(live_ids)]
        if to_start:
            ec2.start_instances(InstanceIds=to_start)
    except Exception:
        pass


def response_poller():
    while True:
        try:
            response = sqs.receive_message(
                QueueUrl=RESP_QUEUE_URL,
                MaxNumberOfMessages=10,
                WaitTimeSeconds=20,
                MessageAttributeNames=["All"],
            )
            for message in response.get("Messages", []):
                request_id = (
                    message.get("MessageAttributes", {})
                    .get("request_id", {})
                    .get("StringValue", "")
                )
                with waiters_lock:
                    waiter = response_waiters.get(request_id)

                if waiter is not None:
                    waiter.put(message.get("Body", ""))

                sqs.delete_message(
                    QueueUrl=RESP_QUEUE_URL,
                    ReceiptHandle=message["ReceiptHandle"],
                )
        except Exception:
            time.sleep(1)


threading.Thread(target=response_poller, daemon=True).start()


def get_cached_result(filename):
    image_name = filename.rsplit(".", 1)[0]
    item = table.get_item(
        Key={"filename": image_name}, ConsistentRead=True
    ).get("Item")
    if item and "result" in item:
        return f"{filename}:{item['result']}"
    return None


@app.route("/", methods=["GET"])
def healthcheck():
    return "web tier running", 200


@app.route("/", methods=["POST"])
def upload():
    if "inputFile" not in request.files:
        return "missing inputFile", 400

    uploaded_file = request.files["inputFile"]
    filename = os.path.basename(uploaded_file.filename or "")
    if not filename:
        return "empty filename", 400

    request_id = str(uuid.uuid4())
    waiter = queue.Queue(maxsize=1)

    with waiters_lock:
        response_waiters[request_id] = waiter

    try:
        cached_result = get_cached_result(filename)
        if cached_result is not None:
            return cached_result, 200

        ensure_app_capacity()
        s3.upload_fileobj(uploaded_file, S3_BUCKET, filename)
        sqs.send_message(
            QueueUrl=REQ_QUEUE_URL,
            MessageBody=filename,
            MessageAttributes={
                "request_id": {"StringValue": request_id, "DataType": "String"}
            },
        )

        result = waiter.get(timeout=120)
        return result, 200
    except queue.Empty:
        return "timeout waiting for app-tier response", 504
    except Exception as exc:
        return f"internal error: {exc}", 500
    finally:
        with waiters_lock:
            response_waiters.pop(request_id, None)
