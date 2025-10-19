from pathlib import Path

import gdown
import loguru

TORCH_WEIGHTS_URLS = {
    "v_48_002.pt": "https://drive.google.com/file/d/1oAEoRANp0kRzH8dduecVQ5nFmpXYpm3y/view?usp=drive_link",
    "v_48_010.pt": "https://drive.google.com/file/d/1DeSbgKRAxrMyjOJMuo1WSU1neRTR2Yqf/view?usp=drive_link",
    "v_48_020.pt": "https://drive.google.com/file/d/1VsL01dMIj0_SFfCXFA4qATod2DZLivLz/view?usp=drive_link",
    "v_48_030.pt": "https://drive.google.com/file/d/1yfYgvNNqZtZKx5kv7JNTuSq3m7QfdFap/view?usp=drive_link",
}
JAX_WEIGHTS_URLS = {
    "v_48_002.eqx": "https://drive.google.com/file/d/1CyXP4vXLZRhi3s6n4puTwnXcqm_3XJf_/view?usp=drive_link",
    "v_48_010.eqx": "https://drive.google.com/file/d/16eKY0FR24bcgU7CJAZbyZTH2Cf7pZuTD/view?usp=drive_link",
    "v_48_020.eqx": "https://drive.google.com/file/d/1mHxOCkAMm449QqHmZljHPvFI7o-baZUr/view?usp=drive_link",
    "v_48_030.eqx": "https://drive.google.com/file/d/1Y4bepyQbRQfW7tMd0YmJRUVpoR3AUycl/view?usp=drive_link",
}

ALL_WEIGHTS_URLS = {**TORCH_WEIGHTS_URLS, **JAX_WEIGHTS_URLS}


weights_dir = Path(__file__).parent.parent / "weights"


def main() -> None:
    weights_dir.mkdir(exist_ok=True)
    for name, url in ALL_WEIGHTS_URLS.items():
        output = weights_dir / name
        if output.exists():
            loguru.logger.info(f"File {output} already exists, skipping download.")
            continue
        loguru.logger.info(f"Downloading {name} weights...")
        gdown.download(url, str(output), quiet=False, fuzzy=True)


if __name__ == "__main__":
    main()
