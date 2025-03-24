import os
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv
from openai import OpenAI


@dataclass
class Message:
    role: str
    content: str


class APIHandler:
    def __init__(self):
        self.api_key = self._load_api_key()
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    @staticmethod
    def _load_api_key() -> str:
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("環境変数 GEMINI_API_KEY が設定されていません。")
        return api_key

    @staticmethod
    def load_system_prompt(file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def call_api(self, system_prompt: str, messages: List[Message]) -> dict:
        """
        APIを呼び出して応答を取得する関数

        Args:
            system_prompt: システムプロンプト
            messages: 会話履歴のリスト
        """
        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend([{"role": m.role, "content": m.content} for m in messages])

        return self.client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=api_messages,
            temperature=0.5,
            stream=False,
        )

    @staticmethod
    def extract_reply_content(response) -> str:
        try:
            return response.choices[0].message.content
        except (KeyError, IndexError) as e:
            raise ValueError(f"返答の解析に失敗しました: {e}")
