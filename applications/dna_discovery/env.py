from __future__ import annotations

import re
import numpy as np
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont

import gymnasium as gym

from core.env_base import BaseEnv
from core.spaces import DiscreteActionSpace, MultiDiscreteSpace

# 0=A, 1=C, 2=G, 3=T, 4=M
BASE_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'M': 4}
INT_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'M'}

TARGET_SEQUENCES = [
    "AGCTGACTAGTC", "ACACACTATGGC", "ACTACGTGTGGT", "AGTGCGATGCGT",
    "ACGATGCGACCA", "AGCTATCCACGA", "ATGCAGCTCAGT", "CACGTGACATGT",
    "ACAGTTGCGCGA", "CACGACAGGCTA", "AGTGTCACGGTG", "CAAGTGAGAGAG",
    "CATCGTATCAAC", "CAGTGTCAGGAC", "ATCTTAGACTGC", "CAGACATTGCGT",
    "CGATGCACCAGA", "CTAGAGACTCTT", "ATGGCAGCTCTA", "CTGAGATACGCG",
    "CCGACTGAGATG", "CCTCTCGTGATC", "CATATCGCAGTT", "CGTGCATTATCA",
    "CTAACGCAGTCA", "CTCAATGACTCA", "ATCGATCTGTGG", "CTCGTGGAGTAG",
    "GCGTTACACACA", "GAACTGTATCTC", "CTGGACTCATAG", "GAGGCTCATCAT",
    "GATACGTCCTGA", "GATTAGCACTCT", "TCCCTTGTCTCC", "ACGAGACTGATT",
    "GCTGTACGGATT", "ATCACCAGGTGT", "TGGTCAACGATA", "ATCGCACAGTAA",
    "GTCGTGTAGCCT", "AGCGGAGGTTAG", "ATCGCACTAGTA", "GTCGTAGCCTAG",
    "AGCGGAGGTTAC", "ACGAGACTGATC", "GCTGTACGGATC", "ATCACCAGGTGC",
    "ATCGCACAGTAC", "GTCGTGTAGCCA", "AGCGGAGGTTAA", "ATCGCACTAGTC",
    "GTCGTAGCCTAC"
]

@lru_cache(maxsize=100000)
def _is_compatible(state_str: str) -> bool:
    pattern = state_str.replace('M', '.')
    regex = re.compile(f"^{pattern}$")
    for seq in TARGET_SEQUENCES:
        if regex.match(seq):
            return True
    return False

class MaskedDNADiscoveryEnv(BaseEnv):
    def __init__(self, config: dict) -> None:
        self.config = config
        self.seq_len = 12
        
        # Space definitions
        self.observation_space = MultiDiscreteSpace([5] * self.seq_len)
        self.action_space = DiscreteActionSpace(self.seq_len * 4) # 48 actions
        
        self.state = None
        self.episode_steps = 0
        
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            np.random.seed(seed)
            
        self.state = np.full(self.seq_len, BASE_TO_INT['M'], dtype=np.int64)
        self.episode_steps = 0
        
        return np.copy(self.state), self._get_info()
        
    def _to_string(self, state_arr: np.ndarray) -> str:
        return "".join(INT_TO_BASE[int(val)] for val in state_arr)
        
    def _get_info(self) -> dict:
        return {
            "symbolic_state": self._to_string(self.state),
            "valid_actions": self.valid_actions(self.state)
        }
        
    def valid_actions(self, state_arr: np.ndarray) -> list[int]:
        valid = []
        for i in range(self.seq_len):
            if state_arr[i] == BASE_TO_INT['M']:
                for b in range(4):
                    valid.append(i * 4 + b)
        return valid

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
            
        self.episode_steps += 1
        index = int(action) // 4
        base_idx = int(action) % 4
        
        # Dead-end: Invalid action (modifying an already filled position)
        if self.state[index] != BASE_TO_INT['M']:
            return np.copy(self.state), -0.1, True, False, self._get_info()
            
        # Apply action
        self.state[index] = base_idx
        state_str = self._to_string(self.state)
        
        # Dead-end: Incompatible state
        if not _is_compatible(state_str):
            return np.copy(self.state), -0.1, True, False, self._get_info()
            
        # Success: All filled and compatible
        if BASE_TO_INT['M'] not in self.state:
            # Mark success in info for eval metrics
            info = self._get_info()
            info["success"] = True
            return np.copy(self.state), 1.0, True, False, info
            
        # Step: Valid and ongoing
        return np.copy(self.state), 0.0, False, False, self._get_info()

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("Only mode='rgb_array' is supported")
            
        width, height = 400, 100
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        
        text = self._to_string(self.state)
        
        font = ImageFont.load_default()
            
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (width - text_width) / 2
        y = (height - text_height) / 2
        
        draw.text((x, y), text, fill="black", font=font)
        
        return np.array(image)

    def close(self) -> None:
        pass
