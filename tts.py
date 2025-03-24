from style_bert_vits2.tts_model import TTSModel
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
import sounddevice as sd
import numpy as np
from pathlib import Path


class TTSHandler:
    """音声合成を管理するクラス"""

    def __init__(self, model_dir: str = "model_assets/Kayoko_3variant"):
        self.model_dir = Path(model_dir)
        print("TTSモデルを初期化しています...")
        self._init_bert()
        self.model = self._init_tts_model()

    def _init_bert(self) -> None:
        """BERTモデルとトークナイザーを初期化"""
        model_name = "ku-nlp/deberta-v2-large-japanese-char-wwm"
        bert_models.load_model(Languages.JP, model_name)
        bert_models.load_tokenizer(Languages.JP, model_name)

    def _init_tts_model(self) -> TTSModel:
        """TTSモデルを初期化"""
        return TTSModel(
            model_path=str(self.model_dir / "Kayoko_3variant_e100_s4600.safetensors"),
            config_path=str(self.model_dir / "config.json"),
            style_vec_path=str(self.model_dir / "style_vectors.npy"),
            device="cuda",
        )

    def generate_and_play(self, text: str) -> None:
        """テキストから音声を生成して再生"""
        sample_rate, audio = self.model.infer(text=text)
        self._play_audio(audio, sample_rate)

    @staticmethod
    def _play_audio(audio: np.ndarray, sample_rate: int) -> None:
        """音声データを再生"""
        audio_float = audio.astype(np.float32)
        audio_normalized = audio_float / np.max(np.abs(audio_float))
        sd.play(audio_normalized, sample_rate)
        sd.wait()
