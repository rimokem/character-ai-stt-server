import numpy as np
from scipy.io.wavfile import write


class AudioProcessor:
    """音声ファイル処理を管理するクラス"""

    @staticmethod
    def save_wav(
        recording: np.ndarray, sample_rate: int, filename: str = "recording.wav"
    ) -> None:
        """録音データをWAVファイルとして保存する"""
        write(filename, sample_rate, recording)
        print(f"音声を {filename} に保存しました")
