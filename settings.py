from dataclasses import dataclass


@dataclass
class Settings:
    api_key: str = 'FVG5GNAP'
    api_host: str = 'http://localhost:10019'
    poll_interval: float = 0.5


settings = Settings()
