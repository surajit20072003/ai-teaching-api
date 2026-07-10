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
    """Upload bytes to B2 and return the public URL.

    Bug #5 fix: use get_running_loop() instead of deprecated get_event_loop().
    Bug #6 fix: wrap the executor call in asyncio.wait_for(timeout=120) so a
                stalled B2 network connection cannot freeze the pregen batch
                indefinitely. Raises RuntimeError on timeout so the caller can
                handle it and mark the slide as failed rather than hanging forever.
    """
    bucket = os.getenv("B2_BUCKET", "simplelecture-media")
    loop = asyncio.get_running_loop()  # Bug #5: was get_event_loop()

    def _upload():
        client = _get_client()
        client.put_object(
            Body=data,
            Bucket=bucket,
            Key=path,
            ContentType=mime
        )

    # Bug #6: enforce a hard 120-second timeout to prevent silent thread stalls
    try:
        await asyncio.wait_for(loop.run_in_executor(None, _upload), timeout=120)
    except asyncio.TimeoutError:
        raise RuntimeError(
            f"B2 upload timed out after 120s — path={path} size={len(data)}B"
        )

    endpoint = os.getenv("B2_ENDPOINT", "https://s3.us-east-005.backblazeb2.com")
    return f"{endpoint}/{bucket}/{path}"
