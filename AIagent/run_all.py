"""一键运行全部课程（离线 Mock）。用于快速自检 / 课堂总览。

    python run_all.py
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

LESSONS = sorted((Path(__file__).parent / "lessons").glob("lesson_*.py"))


def main():
    for path in LESSONS:
        print("\n" + "=" * 70)
        print(f"▶ 运行 {path.name}")
        print("=" * 70)
        try:
            runpy.run_path(str(path), run_name="__main__")
        except Exception as e:  # noqa: BLE001
            print(f"!! {path.name} 运行出错: {e}")
            raise


if __name__ == "__main__":
    main()
    print("\n✅ 全部课程运行完毕。")
