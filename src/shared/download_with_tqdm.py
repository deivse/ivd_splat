from pathlib import Path
import urllib.request
from tqdm import tqdm
from filelock import FileLock


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_with_pbar(url, output_path):
    output_path = Path(output_path)
    lock_path = output_path.with_suffix(output_path.suffix + ".lock")
    with FileLock(lock_path):
        if output_path.exists():
            return
        try:
            with DownloadProgressBar(
                unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
            ) as t:
                urllib.request.urlretrieve(
                    url, filename=output_path, reporthook=t.update_to
                )
        except KeyboardInterrupt:
            print(f"Keyboard interrupt. Deleting incomplete download at {output_path}")
            output_path.unlink()
            raise
