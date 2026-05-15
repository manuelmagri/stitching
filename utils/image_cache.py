"""Cache di frame in RAM: legge una sola volta da disco, applica il downscale e tiene
i frame riencodati JPEG in memoria. Il decode al volo restituisce sempre un array BGR
alla stessa risoluzione che usa il resto della pipeline (feature, warp, mosaico).

La compressione JPEG (lossy) abbatte l'occupazione RAM rispetto al raw uint8 di un
fattore tipico ~10x a qualita' 85, mentre la risoluzione resta invariata.
"""
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


class FrameCache:
    def __init__(
        self,
        paths: list[Path],
        downscale: float = 1.0,
        jpeg_quality: int = 85,
        verbose: bool = True,
    ):
        self._encoded: list[bytes | None] = []
        self._shape: tuple[int, int] | None = None  # (w, h) dopo downscale
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

        total_bytes = 0
        iterator = tqdm(paths, desc="cache") if verbose else paths
        for p in iterator:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                self._encoded.append(None)
                continue
            if downscale != 1.0:
                nw = int(round(img.shape[1] / downscale))
                nh = int(round(img.shape[0] / downscale))
                img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            if self._shape is None:
                self._shape = (img.shape[1], img.shape[0])
            ok, buf = cv2.imencode(".jpg", img, encode_params)
            if not ok:
                self._encoded.append(None)
                continue
            data = buf.tobytes()
            self._encoded.append(data)
            total_bytes += len(data)

        self.jpeg_quality = jpeg_quality
        self.downscale = downscale
        self.total_bytes = total_bytes

        if verbose:
            mb = total_bytes / (1024 * 1024)
            n_ok = sum(1 for b in self._encoded if b is not None)
            print(
                f"[cache] {n_ok}/{len(paths)} frame in RAM, JPEG q={jpeg_quality}, "
                f"downscale={downscale}, totale ~{mb:.1f} MB"
            )

    @property
    def image_size(self) -> tuple[int, int]:
        if self._shape is None:
            raise RuntimeError("cache vuota: nessun frame caricato")
        return self._shape

    def __len__(self) -> int:
        return len(self._encoded)

    def get(self, i: int) -> np.ndarray | None:
        buf = self._encoded[i]
        if buf is None:
            return None
        arr = np.frombuffer(buf, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
