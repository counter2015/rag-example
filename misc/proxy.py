import os
from dotenv import load_dotenv


def setup_proxy():
    load_dotenv()
    http_proxy = os.getenv("HTTP_PROXY")
    https_proxy = os.getenv("HTTPS_PROXY")

    if http_proxy or https_proxy:
        print(f"using proxy:")
        if http_proxy:
            print(f"  HTTP_PROXY: {http_proxy}")
        if https_proxy:
            print(f"  HTTPS_PROXY: {https_proxy}")
    else:
        pass
