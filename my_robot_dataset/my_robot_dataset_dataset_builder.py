import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path


class MyRobotDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}

    def _info(self):
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(shape=(224, 224, 3), dtype=np.uint8),
                        "state": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    }),
                    "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    "is_first": tf.bool,
                    "is_last": tf.bool,
                    "is_terminal": tf.bool,
                    "reward": tf.float32,
                    "language_instruction": tfds.features.Text(),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(),
                    "success": tf.bool,
                }),
            })
        )

    def _split_generators(self, dl_manager):
        hdf5_files = sorted((Path(__file__).parent.parent / "prepared").glob("episode_*.hdf5"))
        print(f"[Builder] Found {len(hdf5_files)} episodes: {hdf5_files}")
        return {"train": self._generate_examples(hdf5_files)}

    def _generate_examples(self, files):
        for fpath in files:
            with h5py.File(fpath, "r") as f:
                images     = f["observations/images"][:]
                states     = f["observations/joint_states"][:]
                actions    = f["ee_actions"][:]
                is_first   = f["is_first"][:]
                is_last    = f["is_last"][:]
                is_term    = f["is_terminal"][:]
                rewards    = f["rewards"][:]
                lang_steps = f["step_language_instruction"][:]
                success    = bool(f["episode_metadata"].attrs["success"])

            T = len(actions)
            steps = []
            for i in range(T):
                steps.append({
                    "observation": {
                        "image": images[i],
                        "state": states[i],
                    },
                    "action":      actions[i],
                    "is_first":    bool(is_first[i]),
                    "is_last":     bool(is_last[i]),
                    "is_terminal": bool(is_term[i]),
                    "reward":      float(rewards[i]),
                    "language_instruction": lang_steps[i].decode("utf-8"),
                })

            yield fpath.stem, {
                "steps": steps,
                "episode_metadata": {"file_path": str(fpath), "success": success},
            }
