from pathlib import Path

import h5py
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class ISPDataset(Dataset):
    """
    Dataset for reading RAW / Y / UV patches from HDF5.

    If the HDF5 file contains precomputed ISP outputs (y_isp, uv_isp),
    they are returned directly, skipping ISP during training.
    """

    def __init__(self, h5_path: str, return_meta: bool = True):
        """
        Args:
            h5_path: path to the HDF5 file with patches
            return_meta: whether to return patch metadata
        """
        self.h5_path = str(h5_path)
        self.return_meta = return_meta
        self._h5 = None

        path = Path(self.h5_path)
        if not path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {path}")

        with h5py.File(path, "r") as h5_file:
            self.has_raw = "raw" in h5_file
            self.has_isp_outputs = "y_isp" in h5_file and "uv_isp" in h5_file

            if self.has_raw:
                self.length = int(h5_file["raw"].shape[0])
            elif self.has_isp_outputs:
                self.length = int(h5_file["y_isp"].shape[0])
            else:
                raise KeyError(
                    "HDF5 dataset must contain either raw patches or "
                    "precomputed y_isp/uv_isp tensors"
                )

    def __len__(self) -> int:
        return self.length

    def _get_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0 or index >= self.length:
            raise IndexError(f"Index {index} is out of range for dataset of length {self.length}")

        h5_file = self._get_h5()

        y_ref = torch.from_numpy(h5_file["y_ref"][index]).float().unsqueeze(0) / 255.0
        uv_ref = torch.from_numpy(h5_file["uv_ref"][index]).float() / 255.0

        sample = {
            "y_ref": y_ref,
            "uv_ref": uv_ref,
        }

        if self.has_raw:
            sample["raw"] = torch.from_numpy(h5_file["raw"][index]).float().unsqueeze(0) / 4095.0

        if self.has_isp_outputs:
            sample["y_isp"] = torch.from_numpy(h5_file["y_isp"][index])
            sample["uv_isp"] = torch.from_numpy(h5_file["uv_isp"][index])

        if self.return_meta and "scene_id" in h5_file:
            sample["scene_id"] = int(h5_file["scene_id"][index])
            sample["frame_idx"] = int(h5_file["frame_idx"][index])
            sample["x"] = int(h5_file["x"][index])
            sample["y"] = int(h5_file["y"][index])

        return sample

    def close(self):
        h5_file = getattr(self, "_h5", None)
        self._h5 = None

        if h5_file is None:
            return

        try:
            h5_file.close()
        except (AttributeError, TypeError, ValueError):
            # During interpreter shutdown h5py internals may already be partially
            # destroyed, so close() can fail even though the handle is no longer needed.
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def create_dataloader(
    h5_path: str,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
    return_meta: bool = True,
    balance_scenes: bool = False,
) -> DataLoader:
    dataset = ISPDataset(h5_path=h5_path, return_meta=return_meta)
    sampler = None

    if balance_scenes:
        with h5py.File(h5_path, "r") as h5_file:
            if "scene_id" not in h5_file:
                raise KeyError("balance_scenes=True requires a 'scene_id' dataset in HDF5")
            scene_ids = torch.as_tensor(h5_file["scene_id"][:], dtype=torch.long)
        unique, counts = torch.unique(scene_ids, return_counts=True)
        inv_counts = {
            int(scene.item()): 1.0 / float(count.item())
            for scene, count in zip(unique, counts, strict=True)
        }
        sample_weights = torch.tensor(
            [inv_counts[int(scene.item())] for scene in scene_ids],
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
