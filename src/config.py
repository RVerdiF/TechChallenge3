
from pathlib import Path

# Project Root
ROOT_DIR = Path(__file__).parent.parent

# Data
DATA_DIR = ROOT_DIR / "src" / "DataHandler"
BTC_PRICES_DB = DATA_DIR / "btc_prices.db"
USERS_DB = DATA_DIR / "users.db"

# User Data
USER_DATA_DIR = ROOT_DIR / "user_data"

# Config
CONFIG_FILE = DATA_DIR / "config.json"
