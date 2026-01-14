from dataclasses import dataclass


@dataclass
class Settings:
    api_key: str = 'QXBFZ5LE'
    api_host: str = 'http://localhost:10020'
    poll_interval: float = 0.5


settings = Settings()
