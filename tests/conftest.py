"""Configuration for the pytest test suite."""

import sys
from pathlib import Path

# Expose scripts/demo/ so tests can import standalone scripts like rule_config
# and configurable_fake_agent without adding them to the installed package.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "demo"))
