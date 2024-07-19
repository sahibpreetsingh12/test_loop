import os
import torch
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger('fastapi')

class SQLModelDownloader:
    def __init__(self):
        from src.core.config import settings  # Import here to avoid circular import
        self.device = settings.HARDWARE_DEVICE
        self.model_name = settings.SQL_MODEL_NAME
        self.sql_model_dir = self._create_model_dir(Path.cwd())
        self.sql_tokenizer_dir = self._create_tokenizer_dir(Path.cwd())
        self._download_tokenizer()
        self._download_model()

    def _create_model_dir(self, project_dir):
        model_dir = project_dir / "sql_model"
        model_dir.mkdir(exist_ok=True)
        logger.info(f"Model directory '{model_dir}' created or already exists.")
        return model_dir

    def _create_tokenizer_dir(self, project_dir):
        tokenizer_dir = project_dir / "sql_tokenizer"
        tokenizer_dir.mkdir(exist_ok=True)
        logger.info(f"Tokenizer directory '{tokenizer_dir}' created or already exists.")
        return tokenizer_dir

    def _download_tokenizer(self):
        if (self.sql_tokenizer_dir / "tokenizer_config.json").exists():
            logger.info(f"SQL Tokenizer found locally at {self.sql_tokenizer_dir}")
            return None
        
        logger.info(f"Tokenizer not found locally. Downloading from {self.model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.save_pretrained(str(self.sql_tokenizer_dir))
        logger.info(f"Tokenizer saved to {self.sql_tokenizer_dir}")
        return tokenizer

    def _download_model(self):
        if (self.sql_model_dir / "config.json").exists():
            logger.info(f"SQL model found locally at {self.sql_model_dir}")
            return None
        
        logger.info(f"Model not found locally. Downloading from {self.model_name}...")
        model = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_4bit=True)
        model.save_pretrained(str(self.sql_model_dir))
        logger.info(f"Model saved to {self.sql_model_dir}")
        return model