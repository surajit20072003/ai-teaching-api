import boto3, os, asyncio
from botocore.config import Config

def _get_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("B2_ENDPOINT", "https://s3.us-east-005.backblazeb2.com"),
        aws_access_key_id=os.getenv("B2_KEY_ID", ""),
        aws_secret_access_key=os.getenv("B2_APP_KEY", ""),
        config=Config(signature_version="s3v4")
    )

async def upload_to_b2(data: bytes, path: str, mime: str = "application/octet-stream") -> str:
    """Upload bytes to B2 and return the public URL."""
    bucket = os.getenv("B2_BUCKET", "simplelecture-media")
    loop = asyncio.get_event_loop()

    def _upload():
        client = _get_client()
        client.put_object(
            Body=data,
            Bucket=bucket,
            Key=path,
            ContentType=mime
        )

    await loop.run_in_executor(None, _upload)
    endpoint = os.getenv("B2_ENDPOINT", "https://s3.us-east-005.backblazeb2.com")
    return f"{endpoint}/{bucket}/{path}"
