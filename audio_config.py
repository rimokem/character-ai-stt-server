from dataclasses import dataclass


@dataclass
class AudioConfig:
    """録音設定を保持するデータクラス"""

    threshold: float = 0.01
    silence_duration: float = 1.0
    max_duration: float = 30
    sample_rate: int = 16000
    chunk_duration: float = 0.1  # 秒単位のチャンクサイズ
