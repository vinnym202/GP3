TRAIN_START_DATE = "2011-01-03" # Start on a Monday
TRAIN_END_DATE = "2023-12-31"

VAL_START_DATE = "2017-01-01"
VAL_END_DATE = "2017-01-31"

TEST_START_DATE = "2024-01-01"
TEST_END_DATE = "2024-04-30"

TRADE_START_DATE = "2023-01-01"
TRADE_END_DATE = "2023-05-16"

ERL_PARAMS = {
    "learning_rate": 1e-5,
    "weight_decay": 1e-4,
    "lambda_entropy": 2e-2,
    "batch_size": 256,
    "gamma":  0.995,
    "net_dimension":[256, 128, 1024],
    "horizon_len": 1000,
    "repeat_times": 8,
    "optimizer": "RAdam",
    "clip_grad_norm": 3.0,
    "ratio_clip": 0.25,
    "lambda_gae_adv": 0.9,
    "activation": "GELU",
    "reward_scaling": 2 ** -11,
    "random_seed":42,
}

# stockstats technical indicator column names
# check https://pypi.org/project/stockstats/ for different names
INDICATORS = [
    "vwma",
    "atr",
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi",
    "cci",
    "mfi",
    "kdjk",
    "kdjd",
    "kdjj",
    "supertrend",
    "tema",
    "ichimoku",
]

# Possible time zones
TIME_ZONE_SHANGHAI = "Asia/Shanghai"  # Hang Seng HSI, SSE, CSI
TIME_ZONE_USEASTERN = "US/Eastern"  # Dow, Nasdaq, SP
TIME_ZONE_PARIS = "Europe/Paris"  # CAC,
TIME_ZONE_BERLIN = "Europe/Berlin"  # DAX, TECDAX, MDAX, SDAX
TIME_ZONE_JAKARTA = "Asia/Jakarta"  # LQ45
TIME_ZONE_SELFDEFINED = "xxx"  # If neither of the above is your time zone, you should define it, and set USE_TIME_ZONE_SELFDEFINED 1.
USE_TIME_ZONE_SELFDEFINED = 0  # 0 (default) or 1 (use the self defined)

# parameters for data sources
ALPACA_API_KEY = "xxx"  # your ALPACA_API_KEY
ALPACA_API_SECRET = "xxx"  # your ALPACA_API_SECRET
ALPACA_API_BASE_URL = "https://paper-api.alpaca.markets"  # alpaca url
