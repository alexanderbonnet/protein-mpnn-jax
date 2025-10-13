from pathlib import Path

import gdown
import loguru

WEIGHTS_URLS = {
    "v_48_002": "https://drive.google.com/file/d/1oAEoRANp0kRzH8dduecVQ5nFmpXYpm3y/view?usp=drive_link",
    "v_48_010": "https://drive.google.com/file/d/1DeSbgKRAxrMyjOJMuo1WSU1neRTR2Yqf/view?usp=drive_link",
    "v_48_020": "https://drive.google.com/file/d/1VsL01dMIj0_SFfCXFA4qATod2DZLivLz/view?usp=drive_link",
    "v_48_030": "https://drive.google.com/file/d/1yfYgvNNqZtZKx5kv7JNTuSq3m7QfdFap/view?usp=drive_link",
}


weights_dir = Path(__file__).parent.parent / "weights"


def main() -> None:
    weights_dir.mkdir(exist_ok=True)
    for name, url in WEIGHTS_URLS.items():
        output = weights_dir / f"{name}.pt"
        if output.exists():
            loguru.logger.info(f"File {output} already exists, skipping download.")
            continue
        loguru.logger.info(f"Downloading {name} weights...")
        gdown.download(url, str(output), quiet=False, fuzzy=True)


if __name__ == "__main__":
    main()
