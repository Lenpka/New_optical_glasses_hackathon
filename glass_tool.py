#!/usr/bin/env python
"""Точка входа: инструмент для восстановления и генерации составов стекла.

Примеры:
    python glass_tool.py check
    python glass_tool.py recover --nd 1.85 --vd 25
    python glass_tool.py recover --glass "N-SF11"
    python glass_tool.py generate --nd 1.90 --vd 22 -n 400 --out candidates.csv
    python glass_tool.py interactive

Веб-интерфейс:
    pip install streamlit
    streamlit run glass_tool_app.py
"""

from glass_tool.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
