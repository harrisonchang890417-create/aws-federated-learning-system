import csv
import os
import time

import boto3

REGION = "us-west-2"
ASUID = "1236295946"
REQ_QUEUE_URL = f"https://sqs.{REGION}.amazonaws.com/671194378296/{ASUID}-req-queue"
RESP_QUEUE_URL = f"https://sqs.{REGION}.amazonaws.com/671194378296/{ASUID}-resp-queue"
DYNAMODB_TABLE = f"{ASUID}-dynamoDB"

PRED_CSV_PATH = "/home/ec2-user/classification_face_images_1000.csv"
if not os.path.exists(PRED_CSV_PATH):
    PRED_CSV_PATH = "classification_face_images_1000.csv"

sqs = boto3.client("sqs", region_name=REGION)
dynamodb = boto3.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(DYNAMODB_TABLE)

prediction_map = {}
with open(PRED_CSV_PATH, newline="") as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        prediction_map[row["Image"]] = row["Results"]


def get_cached_prediction(image_name):
    item = table.get_item(
        Key={"filename": image_name}, ConsistentRead=True
    ).get("Item")
    if item and "result" in item:
        return item["result"]
    return None


def put_prediction(image_name, prediction):
    table.put_item(Item={"filename": image_name, "result": prediction})


while True:
    try:
        response = sqs.receive_message(
            QueueUrl=REQ_QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
            MessageAttributeNames=["All"],
        )

        messages = response.get("Messages", [])
        if not messages:
            continue

        for message in messages:
            filename = message.get("Body", "")
            receipt_handle = message["ReceiptHandle"]
            request_id = (
                message.get("MessageAttributes", {})
                .get("request_id", {})
                .get("StringValue", "")
            )
            image_name = filename.rsplit(".", 1)[0]

            prediction = get_cached_prediction(image_name)
            if prediction is None:
                prediction = prediction_map.get(image_name, "")
                if prediction:
                    put_prediction(image_name, prediction)

            sqs.send_message(
                QueueUrl=RESP_QUEUE_URL,
                MessageBody=f"{filename}:{prediction}",
                MessageAttributes={
                    "request_id": {"StringValue": request_id, "DataType": "String"}
                },
            )
            sqs.delete_message(QueueUrl=REQ_QUEUE_URL, ReceiptHandle=receipt_handle)
    except Exception:
        time.sleep(1)
