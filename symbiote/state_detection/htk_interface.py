"""Interface between Python and the HTK Hidden Markov Model toolkit.

Provides helpers for:
* writing / reading HTK binary feature files (``.mfc``),
* writing HTK label files (``.lab``) and master label files (``.mlf``),
* generating the HMM prototype and configuration files,
* running HTK training (``HCompV``, ``HERest``) and decoding (``HVite``)
  via ``subprocess``.
"""

import os
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DEFAULT_HTK_CONFIG, HTKConfig


# Valid HTK state labels (emitting states only)
_STATES = ["PICK", "CARRY_WITH", "PLACE", "CARRY_EMPTY"]


class HTKStateDetector:
    """HTK-based HMM state detector."""

    def __init__(self, model_dir: str, config: Optional[HTKConfig] = None):
        self.model_dir = model_dir
        self.config = config or DEFAULT_HTK_CONFIG

    # ==================================================================
    # Training
    # ==================================================================

    def train(
        self,
        training_data: List[Tuple[np.ndarray, pd.DataFrame]],
        output_dir: str,
        verbose: bool = True,
    ) -> None:
        """Train HTK HMM from ``(features, annotations)`` pairs.

        Creates the full HTK directory structure under *output_dir* and runs
        ``HCompV`` followed by iterative ``HERest`` re-estimation.
        """
        output_dir = os.path.abspath(output_dir)
        features_dir = os.path.join(output_dir, "features")
        labels_dir = os.path.join(output_dir, "labels")
        models_dir = os.path.join(output_dir, "models")
        config_dir = os.path.join(output_dir, "config")

        for d in [features_dir, labels_dir, models_dir, config_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

        # 1. Write feature files and label files -----------------------
        scp_entries: List[str] = []
        mlf_lines = ["#!MLF!#\n"]
        for i, (features, annotations) in enumerate(training_data):
            mfc_path = os.path.join(features_dir, f"train_{i}.mfc")
            lab_path = os.path.join(labels_dir, f"train_{i}.lab")
            self._write_htk_features(features, mfc_path)
            self._write_htk_labels(annotations, lab_path)
            scp_entries.append(mfc_path)

            # Append to master label file
            mlf_lines.append(f'"*/train_{i}.lab"\n')
            for _, row in annotations.iterrows():
                s = int(row["timestamp_start"] * 1e7)
                e = int(row["timestamp_end"] * 1e7)
                mlf_lines.append(f"{s} {e} {row['state']}\n")
            mlf_lines.append(".\n")

        # Write .scp (script) file
        scp_path = os.path.join(config_dir, "train.scp")
        with open(scp_path, "w") as f:
            f.write("\n".join(scp_entries) + "\n")

        # Write master label file
        mlf_path = os.path.join(labels_dir, "labels.mlf")
        with open(mlf_path, "w") as f:
            f.writelines(mlf_lines)

        # 2. Write configuration and prototype -------------------------
        htk_cfg_path = self._write_htk_config(config_dir)
        proto_path = self._write_proto(config_dir)
        wordlist_path = self._write_wordlist(config_dir)
        grammar_path = self._write_grammar(config_dir)

        # 3. HCompV: compute global mean / variance --------------------
        hmm0_dir = os.path.join(models_dir, "hmm0")
        Path(hmm0_dir).mkdir(exist_ok=True)

        cmd_hcompv = [
            "HCompV", "-T", "1",
            "-C", htk_cfg_path,
            "-S", scp_path,
            "-M", hmm0_dir,
            "-f", "0.01",
            proto_path,
        ]
        if verbose:
            print(f"[HTK] Running HCompV ...")
        result = subprocess.run(cmd_hcompv, capture_output=True, text=True)
        if verbose and result.stdout:
            print(result.stdout[-500:])
        if result.returncode != 0:
            raise RuntimeError(
                f"HCompV failed (exit {result.returncode}):\n{result.stderr}"
            )

        # Build the macro / hmmdefs from HCompV output
        self._build_hmm0(hmm0_dir, proto_path)

        # 4. HERest: Baum-Welch re-estimation --------------------------
        prev_dir = hmm0_dir
        for iteration in range(1, self.config.num_training_iterations + 1):
            iter_dir = os.path.join(models_dir, f"hmm{iteration}")
            Path(iter_dir).mkdir(exist_ok=True)

            cmd_herest = [
                "HERest", "-T", "1",
                "-C", htk_cfg_path,
                "-S", scp_path,
                "-I", mlf_path,
                "-M", iter_dir,
                "-H", os.path.join(prev_dir, "macros"),
                "-H", os.path.join(prev_dir, "hmmdefs"),
            ] + _STATES

            if verbose:
                print(f"[HTK] HERest iteration {iteration} ...")
            result = subprocess.run(cmd_herest, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"HERest iter {iteration} failed (exit {result.returncode}):\n"
                    f"{result.stderr}"
                )
            prev_dir = iter_dir

        # 5. Copy final model to canonical location --------------------
        final_dir = os.path.join(models_dir, "hmm_final")
        Path(final_dir).mkdir(exist_ok=True)
        for fn in ["macros", "hmmdefs"]:
            src = os.path.join(prev_dir, fn)
            dst = os.path.join(final_dir, fn)
            if os.path.exists(src):
                with open(src, "rb") as f_in, open(dst, "wb") as f_out:
                    f_out.write(f_in.read())

        # Copy grammar and wordlist for decoding
        for fn in [grammar_path, wordlist_path]:
            dst = os.path.join(final_dir, os.path.basename(fn))
            with open(fn, "rb") as f_in, open(dst, "wb") as f_out:
                f_out.write(f_in.read())

        if verbose:
            print(f"[HTK] Training complete. Final model: {final_dir}")

    # ==================================================================
    # Decoding
    # ==================================================================

    def decode(
        self,
        features: np.ndarray,
        fps: float,
        frame_numbers: Optional[List[int]] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Decode state sequence using Viterbi (HVite).

        Returns DataFrame with ``[timestamp_start, timestamp_end, state]``.
        """
        final_dir = os.path.join(self.model_dir, "models", "hmm_final")
        if not os.path.isdir(final_dir):
            final_dir = self.model_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            mfc_path = os.path.join(tmpdir, "input.mfc")
            self._write_htk_features(features, mfc_path)

            scp_path = os.path.join(tmpdir, "test.scp")
            with open(scp_path, "w") as f:
                f.write(mfc_path + "\n")

            output_mlf = os.path.join(tmpdir, "output.mlf")

            wordlist_path = os.path.join(final_dir, "wordlist")
            grammar_path = os.path.join(final_dir, "grammar")

            # Build word network from grammar
            net_path = os.path.join(tmpdir, "network.slf")
            hp_cmd = [
                "HParse", grammar_path, net_path,
            ]
            subprocess.run(hp_cmd, capture_output=True, text=True)

            htk_cfg_path = self._write_htk_config(tmpdir)

            cmd = [
                "HVite", "-T", "1",
                "-C", htk_cfg_path,
                "-H", os.path.join(final_dir, "macros"),
                "-H", os.path.join(final_dir, "hmmdefs"),
                "-S", scp_path,
                "-i", output_mlf,
                "-w", net_path,
                wordlist_path,
                wordlist_path,
            ]
            if verbose:
                print("[HTK] Running HVite ...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"HVite failed (exit {result.returncode}):\n{result.stderr}"
                )

            segments = self._parse_mlf(output_mlf, fps, features.shape[0])

        return segments

    # ==================================================================
    # HTK file I/O helpers
    # ==================================================================

    def _write_htk_features(
        self, features: np.ndarray, output_path: str
    ) -> str:
        """Write features to HTK binary ``.mfc`` format."""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        n_frames, n_features = features.shape
        sample_period = self.config.sample_period
        samp_size = n_features * 4  # 4 bytes per float32

        header = struct.pack(">IIHH", n_frames, sample_period, samp_size, 9)
        with open(output_path, "wb") as f:
            f.write(header)
            for row in features.astype(np.float32):
                f.write(struct.pack(f">{n_features}f", *row))
        return output_path

    @staticmethod
    def _read_htk_features(path: str) -> np.ndarray:
        """Read an HTK ``.mfc`` file back into a numpy array."""
        with open(path, "rb") as f:
            n_frames, sample_period, samp_size, parm_kind = struct.unpack(
                ">IIHH", f.read(12)
            )
            n_features = samp_size // 4
            data = []
            for _ in range(n_frames):
                row = struct.unpack(f">{n_features}f", f.read(samp_size))
                data.append(row)
        return np.array(data, dtype=np.float32)

    @staticmethod
    def _write_htk_labels(
        annotations: pd.DataFrame, output_path: str
    ) -> None:
        """Write annotations to an HTK ``.lab`` file."""
        with open(output_path, "w") as f:
            for _, row in annotations.iterrows():
                s = int(row["timestamp_start"] * 1e7)
                e = int(row["timestamp_end"] * 1e7)
                f.write(f"{s} {e} {row['state']}\n")

    # ------------------------------------------------------------------
    # Config / prototype generation
    # ------------------------------------------------------------------

    def _write_htk_config(self, output_dir: str) -> str:
        """Write minimal HTK configuration file."""
        path = os.path.join(output_dir, "htk.cfg")
        with open(path, "w") as f:
            f.write("TARGETKIND = USER\n")
            f.write(f"TARGETRATE = {self.config.sample_period}.0\n")
        return path

    def _write_proto(self, output_dir: str) -> str:
        """Write HMM prototype file (single-model topology).

        Each state label gets its own single-state HMM so HTK can
        concatenate them for the cyclic grammar.
        """
        dim = self.config.feature_dim
        mean_str = " ".join(["0.0"] * dim)
        var_str = " ".join(["1.0"] * dim)
        path = os.path.join(output_dir, "proto")

        with open(path, "w") as f:
            f.write(f"~o <VECSIZE> {dim} <USER>\n")
            f.write(f'~h "proto"\n')
            f.write("<BEGINHMM>\n")
            f.write("<NUMSTATES> 3\n")  # entry + 1 emitting + exit
            f.write("<STATE> 2\n")
            f.write(f"<MEAN> {dim}\n  {mean_str}\n")
            f.write(f"<VARIANCE> {dim}\n  {var_str}\n")
            f.write("<TRANSP> 3\n")
            f.write("  0.0 1.0 0.0\n")
            f.write("  0.0 0.6 0.4\n")
            f.write("  0.0 0.0 0.0\n")
            f.write("<ENDHMM>\n")
        return path

    @staticmethod
    def _write_wordlist(output_dir: str) -> str:
        """Write HTK word list (one state label per line)."""
        path = os.path.join(output_dir, "wordlist")
        with open(path, "w") as f:
            for state in _STATES:
                f.write(state + "\n")
        return path

    @staticmethod
    def _write_grammar(output_dir: str) -> str:
        """Write an HTK grammar enforcing cyclic state transitions.

        ``$cycle = PICK CARRY_WITH PLACE CARRY_EMPTY;``
        ``( { $cycle } )``

        This allows one or more full cycles.
        """
        path = os.path.join(output_dir, "grammar")
        with open(path, "w") as f:
            f.write("$cycle = PICK CARRY_WITH PLACE CARRY_EMPTY;\n")
            f.write("( { $cycle } )\n")
        return path

    # ------------------------------------------------------------------
    # Post-HCompV model building
    # ------------------------------------------------------------------

    def _build_hmm0(self, hmm0_dir: str, proto_path: str) -> None:
        """Create per-state HMM definitions from the prototype.

        After ``HCompV`` runs on the single prototype, we replicate it for
        each state label so HTK can train them independently.
        """
        # Read the HCompV-updated proto
        vfloors_path = os.path.join(hmm0_dir, "vFloors")
        proto_out = os.path.join(hmm0_dir, "proto")
        if not os.path.exists(proto_out):
            proto_out = proto_path

        with open(proto_out, "r") as f:
            proto_text = f.read()

        # Extract the ~o header and state body
        lines = proto_text.strip().split("\n")
        header_line = ""
        body_lines = []
        in_body = False
        for line in lines:
            if line.startswith("~o"):
                header_line = line
            elif line.startswith("~h"):
                in_body = True
            elif in_body:
                body_lines.append(line)

        body_text = "\n".join(body_lines)

        # Write macros file (header + vFloors)
        macros_path = os.path.join(hmm0_dir, "macros")
        with open(macros_path, "w") as f:
            f.write(header_line + "\n")
            if os.path.exists(vfloors_path):
                with open(vfloors_path, "r") as vf:
                    f.write(vf.read())

        # Write hmmdefs: one HMM per state
        hmmdefs_path = os.path.join(hmm0_dir, "hmmdefs")
        with open(hmmdefs_path, "w") as f:
            for state_name in _STATES:
                f.write(f'~h "{state_name}"\n')
                f.write(body_text + "\n")

    # ------------------------------------------------------------------
    # MLF parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_mlf(
        mlf_path: str, fps: float, n_frames: int
    ) -> pd.DataFrame:
        """Parse HTK Master Label File output into a DataFrame."""
        segments: List[Dict] = []
        with open(mlf_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith('"') or line == ".":
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    start_100ns = int(parts[0])
                    end_100ns = int(parts[1])
                    state = parts[2]
                    if state in _STATES:
                        segments.append({
                            "timestamp_start": start_100ns / 1e7,
                            "timestamp_end": end_100ns / 1e7,
                            "state": state,
                        })

        if not segments:
            return pd.DataFrame(
                columns=["timestamp_start", "timestamp_end", "state"]
            )
        return pd.DataFrame(segments)


__all__ = ["HTKStateDetector"]
