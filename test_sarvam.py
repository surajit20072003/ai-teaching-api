import os, base64, asyncio
from dotenv import load_dotenv
load_dotenv()

from core.tts_client import synthesize

async def test():
    long_text = "Hello world! " * 50
    b64 = await synthesize(long_text, "hi-IN")
    data = base64.b64decode(b64)
    print("Length:", len(data))
    print("Header:", data[:44])
    try:
        sample_rate  = int.from_bytes(data[24:28], "little")
        channels     = int.from_bytes(data[22:24], "little")
        bits         = int.from_bytes(data[34:36], "little")
        byte_rate    = sample_rate * channels * (bits // 8)
        data_size    = int.from_bytes(data[40:44], "little")
        duration     = data_size / byte_rate if byte_rate else 0
        print("Duration:", duration)
    except Exception as e:
        print("Error:", repr(e))

asyncio.run(test())
