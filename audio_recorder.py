import time
from typing import Tuple, List
import numpy as np
import sounddevice as sd
from audio_config import AudioConfig


class AudioRecorder:
    """音声録音を管理するクラス"""

    def __init__(self, config: AudioConfig):
        self.config = config
        self.is_recording = False

    def record(self) -> Tuple[np.ndarray, int]:
        """音声録音を実行する"""
        print("音声入力を待機中...")
        # 録音開始時にバッファを初期化
        self.audio_buffer: List[np.ndarray] = []
        self.is_recording = False

        try:
            with sd.InputStream(
                samplerate=self.config.sample_rate, channels=1, dtype=np.float32
            ) as stream:
                return self._record_stream(stream)
        except Exception as e:
            print(f"録音中にエラーが発生しました: {e}")
            return np.array([]), self.config.sample_rate

    def _record_stream(self, stream: sd.InputStream) -> Tuple[np.ndarray, int]:
        start_time = last_sound_time = time.time()
        chunk_size = int(self.config.sample_rate * self.config.chunk_duration)

        while True:
            data, overflowed = stream.read(chunk_size)
            current_time = time.time()
            volume_norm = np.linalg.norm(data) / np.sqrt(len(data))

            should_break, last_sound_time = self._handle_recording(
                data, volume_norm, current_time, last_sound_time, start_time
            )
            if should_break:
                break

            if overflowed:
                print("警告: オーバーフローが発生しました")

        return self._process_recording()

    def _handle_recording(
        self,
        data: np.ndarray,
        volume_norm: float,
        current_time: float,
        last_sound_time: float,
        start_time: float,
    ) -> tuple[bool, float]:
        # 録音開始時の処理
        if not self.is_recording and volume_norm > self.config.threshold:
            print("録音を開始します...")
            self.is_recording = True
            self.audio_buffer.append(data.copy())
            return False, current_time

        # 録音中の処理
        if self.is_recording:
            self.audio_buffer.append(data.copy())
            # 音量が閾値を超えた時は最終音声検出時刻を更新
            if volume_norm > self.config.threshold:
                last_sound_time = current_time

            elapsed_silence = current_time - last_sound_time
            total_duration = current_time - start_time

            # 無音判定
            if elapsed_silence >= self.config.silence_duration:
                print("無音を検知したため録音を終了します")
                return True, last_sound_time

            # 最大録音時間の判定
            if total_duration >= self.config.max_duration:
                print("最大録音時間に達しました")
                return True, last_sound_time

        return False, last_sound_time

    def _process_recording(self) -> Tuple[np.ndarray, int]:
        if not self.audio_buffer:
            print("録音データがありません")
            return np.array([]), self.config.sample_rate

        recording = np.concatenate(self.audio_buffer)
        recording = (recording * 32767).astype(np.int16)
        print("録音が完了しました")
        return recording, self.config.sample_rate
