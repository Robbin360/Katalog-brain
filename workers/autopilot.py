"""
Standalone Auto-Pilot worker. Run separately from the web server:
    python -m workers.autopilot

This prevents duplicate patrol loops when the API has multiple replicas.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.worker import auto_pilot_patrol


async def main():
    print("🤖 [Auto-Pilot] Standalone worker started.")
    await auto_pilot_patrol()


if __name__ == "__main__":
    asyncio.run(main())
