"""读取“附件3-模态缺失特征样本/对齐版本”中的 PKL 文件。

默认用法：
    python read_pkl_data.py

读取指定目录：
    python read_pkl_data.py --data-dir "E:\\myProject\\postgraduate\\Coding\\CPMCM\\E_workspace\\00_raw_readonly\\E题数据\\附件3-模态缺失特征样本\\对齐版本"

读取单个文件并查看详细信息：
    python read_pkl_data.py --file "附件3_01.pkl" --show-values

注意：pickle 只能读取可信来源的文件，不要对不可信的 pkl 文件直接执行本脚本。
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any


DEFAULT_DATA_DIR = Path(
    r"E:\myProject\postgraduate\Coding\CPMCM\E_workspace\00_raw_readonly"
    r"\E题数据\附件3-模态缺失特征样本\对齐版本"
)


def load_pkl(path: str | Path) -> Any:
    """读取一个 pkl 文件并返回其中保存的 Python 对象。"""
    path = Path(path)
    with path.open("rb") as file:
        return pickle.load(file)


def load_all(data_dir: str | Path = DEFAULT_DATA_DIR) -> dict[str, Any]:
    """读取目录下全部 pkl 文件，返回 {文件名: 对象}。"""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(f"数据目录不存在：{data_dir}")

    files = sorted(data_dir.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"目录中没有找到 .pkl 文件：{data_dir}")

    result: dict[str, Any] = {}
    for path in files:
        result[path.name] = load_pkl(path)
    return result


def describe(value: Any, show_values: bool = False, indent: int = 0) -> None:
    """递归打印对象的类型、键、形状等信息。"""
    prefix = " " * indent

    if isinstance(value, dict):
        print(f"{prefix}dict ({len(value)} 项)")
        for key, item in value.items():
            print(f"{prefix}{key!r}:")
            describe(item, show_values=show_values, indent=indent + 2)
        return

    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        text = f"{prefix}{type(value).__name__}, shape={shape}"
        if dtype is not None:
            text += f", dtype={dtype}"
        print(text)
        if show_values:
            print(f"{prefix}值预览：{repr(value)[:500]}")
        return

    if isinstance(value, (list, tuple)):
        print(f"{prefix}{type(value).__name__}, length={len(value)}")
        if show_values:
            print(f"{prefix}值预览：{repr(value)[:500]}")
        return

    print(f"{prefix}{type(value).__name__}: {repr(value)[:500]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="批量读取并检查 pkl 特征文件")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="pkl 文件所在目录（默认使用题目附件目录）",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="只读取指定文件名，例如：附件3_01.pkl；也可以传入完整路径",
    )
    parser.add_argument(
        "--show-values",
        action="store_true",
        help="额外打印数组或对象的前 500 个字符",
    )
    args = parser.parse_args()

    if args.file:
        file_path = Path(args.file)
        if not file_path.is_absolute():
            file_path = args.data_dir / file_path
        obj = load_pkl(file_path)
        print(f"文件：{file_path}")
        describe(obj, show_values=args.show_values)
        return

    all_data = load_all(args.data_dir)
    print(f"目录：{args.data_dir}")
    print(f"共读取 {len(all_data)} 个 pkl 文件\n")
    for name, obj in all_data.items():
        print(f"===== {name} =====")
        describe(obj, show_values=args.show_values, indent=2)


if __name__ == "__main__":
    main()
