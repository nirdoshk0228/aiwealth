"""Deterministic risk engine.

Rules: must never import from `app.ai`, and must not call the network or an LLM.
Inputs come in as plain data; outputs are numbers the AI layer may only report.
"""
