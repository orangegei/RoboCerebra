# $ cd /home/shenhaotian/grad/RoboCerebra
# $ python grad/scripts/hf_retry_download.py
import os
import time


from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

REPO_ID = "qiukingballball/RoboCerebra"
LOCAL_DIR = "./RoboCerebra"
REPO_TYPE = "dataset"

MAX_RETRIES = 10
BASE_DELAY = 5  # seconds
MAX_DELAY = 300  # seconds
MAX_WORKERS = 2  # Reduce concurrency to avoid 429


def _sleep(attempt: int, msg: str) -> None:
    delay = min(MAX_DELAY, BASE_DELAY * (2 ** (attempt - 1)))
    print(f"⚠️ {msg}，{delay}s 后重试")
    time.sleep(delay)

def main():
    api = HfApi()
    attempt = 0
    while True:
        try:
            files = api.list_repo_files(REPO_ID, repo_type=REPO_TYPE)
            path = snapshot_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                local_dir=LOCAL_DIR,
                resume_download=True,  # 断点续传
                max_workers=MAX_WORKERS,
            )
            missing = [
                file
                for file in files
                if not os.path.exists(os.path.join(LOCAL_DIR, file))
            ]
            if missing:
                attempt += 1
                if attempt > MAX_RETRIES:
                    raise RuntimeError(f"仍有缺失文件: {len(missing)}")
                _sleep(attempt, f"仍有缺失文件 {len(missing)}")
                continue

            print(f"✅ 下载完成: {path}")
            break
        except HfHubHTTPError as e:
            attempt += 1
            if attempt > MAX_RETRIES:
                raise
            _sleep(attempt, f"下载失败({attempt}/{MAX_RETRIES}): {e}")

if __name__ == "__main__":
    main()