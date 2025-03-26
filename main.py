import logging
from typing import Tuple
from dataclasses import dataclass
import numpy as np
import time
import requests
from fastapi import FastAPI
import uvicorn
from threading import Thread

from audio_config import AudioConfig
from audio_recorder import AudioRecorder
from audio_transcription import TranscriptionHandler

is_up = True
app = FastAPI()


@app.post("/start")
async def start_recording():
    global is_up
    is_up = True
    return {"status": "recording started"}


@app.post("/stop")
async def stop_recording():
    global is_up
    is_up = False
    return {"status": "recording stopped"}


@dataclass
class RecordingSystem:
    """録音と文字起こしシステムのコンポーネントを管理するクラス"""

    recorder: AudioRecorder
    transcriber: TranscriptionHandler

    @classmethod
    def initialize(cls) -> "RecordingSystem":
        """システムの初期化"""
        logging.info("録音と文字起こしシステムを初期化しています...")
        config = AudioConfig()
        return cls(
            recorder=AudioRecorder(config),
            transcriber=TranscriptionHandler(),
        )

    def record_audio(self) -> Tuple[np.ndarray, int]:
        """音声を録音"""
        logging.info("\n音声入力を待機しています...")
        return self.recorder.record()

    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """音声データを文字起こし（一時ファイルを使用）"""
        import tempfile
        import os
        from scipy.io.wavfile import write

        # 一時ファイルを作成して音声データを書き込み
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # 一時ファイルに音声データを書き込み
            write(temp_path, sample_rate, audio_data)

            # 文字起こしを実行
            logging.info("音声を文字起こししています...")
            transcription = self.transcriber.transcribe(temp_path)
            logging.info(f"文字起こし結果:\n{transcription}")

            # 文字起こし結果を外部サーバーに送信
            try:
                response = requests.post(
                    "http://localhost:4114/transcript",
                    json={"text": transcription},
                    timeout=5,
                )
                if response.status_code == 200:
                    logging.info("文字起こし結果を外部サーバーに送信しました")
                else:
                    logging.error(
                        f"外部サーバーへの送信に失敗しました: {response.status_code}"
                    )
            except requests.exceptions.RequestException as e:
                logging.error(f"外部サーバーへの接続に失敗しました: {e}")

            return transcription
        finally:
            # 一時ファイルを削除
            if os.path.exists(temp_path):
                os.remove(temp_path)


def main() -> None:
    # ログ設定
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # システムの初期化
    system = RecordingSystem.initialize()

    # FastAPIサーバーを別スレッドで起動
    Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=4113), daemon=True
    ).start()

    # メインループ
    while True:
        if not is_up:
            time.sleep(1)
            continue
        try:
            # 録音の実行
            logging.info("\n録音を開始します...")
            recording, sample_rate = system.record_audio()

            # 録音データが存在する場合のみ処理を継続
            if recording.any() and is_up:
                # 文字起こしの実行（録音データは保存せず直接処理）
                system.transcribe_audio(recording, sample_rate)
                logging.info("次の録音の準備ができました。")
            else:
                logging.info("録音データがありません。スキップします。")

        except KeyboardInterrupt:
            logging.info("\n\n録音を終了します。\n")
            break
        except Exception as e:
            logging.error(f"\nエラーが発生しました: {e}")
            logging.info("録音を継続します。")


if __name__ == "__main__":
    main()
