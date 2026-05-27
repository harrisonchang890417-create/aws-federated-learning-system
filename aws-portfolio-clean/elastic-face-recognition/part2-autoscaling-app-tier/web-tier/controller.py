import time

import boto3

REGION = "us-west-2"
ASUID = "1236295946"
APP_TIER_NAME_PREFIX = "app-tier-instance"
LAUNCH_TEMPLATE_ID = "lt-0d85c05215d88d238"
LAUNCH_TEMPLATE_VERSION = "$Default"
REQ_QUEUE_URL = f"https://sqs.{REGION}.amazonaws.com/671194378296/{ASUID}-req-queue"
RESP_QUEUE_URL = f"https://sqs.{REGION}.amazonaws.com/671194378296/{ASUID}-resp-queue"
MAX_INSTANCES = 15
POLL_SECONDS = 0.2
KEEP_ALIVE_SECONDS = 4

ec2 = boto3.client("ec2", region_name=REGION)
sqs = boto3.client("sqs", region_name=REGION)


def get_queue_depth(queue_url):
    attrs = sqs.get_queue_attributes(
        QueueUrl=queue_url,
        AttributeNames=[
            "ApproximateNumberOfMessages",
            "ApproximateNumberOfMessagesNotVisible",
        ],
    )["Attributes"]
    visible = int(attrs.get("ApproximateNumberOfMessages", "0"))
    inflight = int(attrs.get("ApproximateNumberOfMessagesNotVisible", "0"))
    return visible + inflight


def list_app_instances():
    reservations = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Name", "Values": [f"{APP_TIER_NAME_PREFIX}*"]},
            {
                "Name": "instance-state-name",
                "Values": ["pending", "running", "stopping", "stopped"],
            },
        ]
    )["Reservations"]

    instances = []
    for reservation in reservations:
        for instance in reservation["Instances"]:
            instances.append(instance)
    return instances


def list_live_instance_ids():
    return [
        instance["InstanceId"]
        for instance in list_app_instances()
        if instance["State"]["Name"] in {"pending", "running"}
    ]


def scale_out():
    instances = list_app_instances()
    live_ids = [
        instance["InstanceId"]
        for instance in instances
        if instance["State"]["Name"] in {"pending", "running"}
    ]
    stopped_ids = [
        instance["InstanceId"]
        for instance in instances
        if instance["State"]["Name"] == "stopped"
    ]

    if len(live_ids) >= MAX_INSTANCES:
        return

    missing = MAX_INSTANCES - len(live_ids)
    if stopped_ids:
        to_start = stopped_ids[:missing]
        ec2.start_instances(InstanceIds=to_start)
        missing -= len(to_start)

    if missing > 0:
        ec2.run_instances(
            MinCount=missing,
            MaxCount=missing,
            LaunchTemplate={
                "LaunchTemplateId": LAUNCH_TEMPLATE_ID,
                "Version": LAUNCH_TEMPLATE_VERSION,
            },
            TagSpecifications=[
                {
                    "ResourceType": "instance",
                    "Tags": [{"Key": "Name", "Value": APP_TIER_NAME_PREFIX}],
                }
            ],
        )


def scale_in():
    live_ids = [
        instance["InstanceId"]
        for instance in list_app_instances()
        if instance["State"]["Name"] == "running"
    ]
    if live_ids:
        ec2.stop_instances(InstanceIds=live_ids)


if __name__ == "__main__":
    keep_alive_until = 0.0
    last_live_count = 0

    while True:
        try:
            req_depth = get_queue_depth(REQ_QUEUE_URL)
            resp_depth = get_queue_depth(RESP_QUEUE_URL)
            live_count = len(list_live_instance_ids())

            if req_depth > 0:
                keep_alive_until = time.time() + KEEP_ALIVE_SECONDS
                scale_out()
            else:
                if live_count > 0 and last_live_count == 0:
                    keep_alive_until = max(
                        keep_alive_until, time.time() + KEEP_ALIVE_SECONDS
                    )
                if resp_depth == 0 and time.time() >= keep_alive_until:
                    scale_in()
            last_live_count = live_count
        except Exception:
            pass

        time.sleep(POLL_SECONDS)
