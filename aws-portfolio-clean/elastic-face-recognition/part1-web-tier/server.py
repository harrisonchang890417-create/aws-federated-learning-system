#!/usr/bin/env python3
"""
CSE546 Project 1 Part I - Web Tier
Face recognition web service using S3 and DynamoDB
ASU ID: 1236295946
"""

import os
import boto3
from flask import Flask, request, Response

# Configuration - US-West-2 region, ASU ID based naming
ASU_ID = "1236295946"
AWS_REGION = "us-west-2"
INPUT_BUCKET = f"{ASU_ID}-in-bucket"
DYNAMODB_TABLE = f"{ASU_ID}-dynamoDB"

app = Flask(__name__)

# Initialize AWS clients
s3_client = boto3.client('s3', region_name=AWS_REGION)
dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)


def get_prediction(filename: str) -> str:
    """Look up face recognition result from DynamoDB."""
    # Use filename without extension as key (CSV uses test_000, not test_000.jpg)
    base_name = os.path.splitext(filename)[0]
    table = dynamodb.Table(DYNAMODB_TABLE)
    response = table.get_item(Key={'filename': base_name})
    return response.get('Item', {}).get('prediction', 'Unknown')


@app.route('/', methods=['POST'])
def face_recognition():
    """Handle POST request with image file, store in S3, return prediction."""
    if 'inputFile' not in request.files:
        return Response("Error: No inputFile in request", status=400, mimetype='text/plain')

    file = request.files['inputFile']
    if file.filename == '':
        return Response("Error: No file selected", status=400, mimetype='text/plain')

    filename = file.filename

    try:
        # Store image in S3 input bucket
        s3_client.upload_fileobj(file, INPUT_BUCKET, filename)

        # Get prediction from DynamoDB
        prediction = get_prediction(filename)

        # Return result in required format: filename:prediction
        return Response(f"{filename}:{prediction}", status=200, mimetype='text/plain')

    except Exception as e:
        return Response(f"Error: {str(e)}", status=500, mimetype='text/plain')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded=True)
