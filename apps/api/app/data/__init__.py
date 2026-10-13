"""Market data providers and caching.

External fetches (mfapi.in, yfinance) run offline/scheduled and are cached;
user requests read from the cache only.
"""
