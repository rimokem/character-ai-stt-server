from faster_whisper import WhisperModel


class TranscriptionHandler:
    """文字起こしを管理するクラス"""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cuda",
        compute_type: str = "int8",
    ):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(
        self,
        audio_path: str,
        initial_prompt: str = "これは「カヨコ」という相手との会話です。",
    ) -> str:
        """音声ファイルを文字起こしする"""
        segments, _ = self.model.transcribe(
            audio_path, language="ja", initial_prompt=initial_prompt
        )
        return " ".join([segment.text for segment in segments])
