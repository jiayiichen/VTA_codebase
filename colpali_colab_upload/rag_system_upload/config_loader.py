"""
Secure configuration loader.

This module loads API keys and credentials from config.json which is gitignored.
The API key is only used when explicitly requested by your code, not automatically.
"""

import json
import os


def load_config(config_path='config.json'):
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to config.json file

    Returns:
        Dictionary with configuration values

    Raises:
        FileNotFoundError: If config.json doesn't exist
        ValueError: If required keys are missing
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Please create it from config.json.template"
        )

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Validate required keys
    required_keys = ['openai_api_key']
    missing_keys = [key for key in required_keys if key not in config]

    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")

    return config


def get_openai_key(config_path='config.json'):
    """
    Get OpenAI API key from config.
    Only called when explicitly needed.
    """
    config = load_config(config_path)
    return config['openai_api_key']


def get_edstem_credentials(config_path='config.json'):
    """
    Get EdStem credentials from config.
    Only called when explicitly needed.
    """
    config = load_config(config_path)
    return {
        'email': config.get('edstem_email'),
        'password': config.get('edstem_password')
    }
