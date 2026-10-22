"""
实验数据处理工具

提供 CSV 加载与统计摘要、Python 代码沙箱执行等功能。
被 DataExecutor 和 CodeExecutor 调用。
"""
import subprocess
import tempfile
import os
from langchain_core.tools import tool


@tool
def load_csv(filepath: str) -> str:
    """加载 CSV 文件并返回详细统计摘要。

    读取CSV文件，返回列名、行数、每列的数据类型、
    数值列的基本统计量（均值、标准差、最大最小值）等。

    Args:
        filepath: CSV 文件路径

    Returns:
        包含行数、列名、数据类型、基本统计信息的字符串
    """
    try:
        import pandas as pd
    except ImportError:
        return "[Error] 请安装 pandas: pip install pandas"

    if not os.path.exists(filepath):
        return f"[Error] 文件不存在: {filepath}"

    try:
        df = pd.read_csv(filepath)

        lines = [
            f"[数据分析] {os.path.basename(filepath)}",
            f"  总行数: {len(df)}",
            f"  总列数: {len(df.columns)}",
            f"  列名: {list(df.columns)}",
            "",
            "--- 数据类型 ---",
        ]
        for col in df.columns:
            lines.append(f"  {col}: {df[col].dtype}")

        # 缺失值统计
        lines.append("")
        lines.append("--- 缺失值统计 ---")
        null_counts = df.isnull().sum()
        has_nulls = False
        for col, cnt in null_counts.items():
            if cnt > 0:
                lines.append(f"  {col}: {cnt} 个缺失 ({cnt / len(df) * 100:.1f}%)")
                has_nulls = True
        if not has_nulls:
            lines.append("  无缺失值")

        # 数值列统计
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            lines.append("")
            lines.append("--- 数值列统计 ---")
            stats = df[numeric_cols].describe()
            lines.append(stats.to_string())

        # 分类列预览
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        if cat_cols:
            lines.append("")
            lines.append("--- 分类列预览 ---")
            for col in cat_cols:
                unique_vals = df[col].nunique()
                top_vals = df[col].value_counts().head(5).to_dict()
                lines.append(f"  {col}: {unique_vals} 个唯一值")
                lines.append(f"    前5: {top_vals}")

        # 数据预览（前5行）
        lines.append("")
        lines.append("--- 数据预览（前5行）---")
        lines.append(df.head().to_string())

        return "\n".join(lines)
    except Exception as e:
        return f"[Error] CSV分析失败 {filepath}: {e}"


@tool
def run_code(code: str) -> str:
    """在子进程中执行可信 Python 代码并返回输出，并非安全沙箱。

    用于运行实验代码（如决策树分类、神经网络训练等）。
    将代码写入临时文件，用 subprocess 执行，捕获 stdout/stderr。
    超时限制 120 秒，防止死循环。

    Args:
        code: 完整可执行的 Python 代码字符串

    Returns:
        代码执行的标准输出。若出错则返回错误信息。
    """
    # 写入临时文件
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, encoding='utf-8'
        ) as f:
            f.write(code)
            tmp_path = f.name

        # 子进程和超时不提供文件、网络或系统权限隔离。
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.path.dirname(tmp_path)
        )

        output_parts = []
        if result.stdout:
            output_parts.append(f"[stdout]\n{result.stdout}")
        if result.stderr:
            output_parts.append(f"[stderr]\n{result.stderr}")
        if not output_parts:
            output_parts.append("[代码执行] 无输出")

        return "\n".join(output_parts)

    except subprocess.TimeoutExpired:
        return "[Error] 代码执行超时（>120秒）"
    except FileNotFoundError:
        return "[Error] 未找到 Python 解释器"
    except Exception as e:
        return f"[Error] 代码执行失败: {e}"
    finally:
        # 清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@tool
def read_code(path: str) -> str:
    """读取用户代码文件或目录中的代码（逐行带行号，含子目录标题）。

    设计目标：
    1) 目录模式下递归读取子目录；
    2) 输出目录标题与目录内文件清单，便于结构化分析；
    3) 每个文件按“每一行”输出，带行号，不截断。

    Args:
        path: 代码文件路径或项目目录路径

    Returns:
        结构化代码文本（目录标题 + 文件标题 + 带行号代码）。
    """
    code_exts = {
        ".py", ".ipynb", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp",
        ".h", ".hpp", ".go", ".rs", ".sh", ".ps1", ".sql", ".yaml", ".yml", ".json",
        ".toml", ".ini", ".cfg", ".md",
    }

    def _read_text_file(fpath: str) -> str:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(fpath, "r", encoding="gbk") as f:
                    return f.read()
            except Exception as e:
                return f"[读取失败] {fpath}: {e}"
        except Exception as e:
            return f"[读取失败] {fpath}: {e}"

    def _with_line_numbers(content: str) -> str:
        lines = content.splitlines()
        if not lines:
            return "   1|"
        width = max(4, len(str(len(lines))))
        return "\n".join(f"{i:>{width}}| {line}" for i, line in enumerate(lines, 1))

    if not os.path.exists(path):
        return f"[Error] 路径不存在: {path}"

    # 单文件模式
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext and ext not in code_exts:
            return f"[CodeReader] 文件存在但不是代码文件类型: {path}"
        raw = _read_text_file(path)
        numbered = _with_line_numbers(raw)
        fname = os.path.basename(path)
        return (
            f"[CodeReader] 读取 1 个文件\n"
            f"## 文件: {fname}\n"
            f"路径: {path}\n"
            f"```text\n{numbered}\n```"
        )

    # 目录模式
    root = os.path.abspath(path)
    dir_map = {}
    file_count = 0

    for cur_root, dirs, files in os.walk(root):
        dirs[:] = sorted([d for d in dirs if not d.startswith(".") and d != "__pycache__"])
        code_files = []
        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            if ext in code_exts:
                code_files.append(os.path.join(cur_root, f))
        if code_files:
            rel_dir = os.path.relpath(cur_root, root)
            dir_key = "." if rel_dir == "." else rel_dir.replace("\\", "/")
            dir_map[dir_key] = code_files
            file_count += len(code_files)

    if not dir_map:
        return f"[CodeReader] 未找到代码文件: {path}"

    sections = [f"[CodeReader] 读取 {file_count} 个代码文件（递归）", f"根目录: {root}", ""]

    for dir_key in sorted(dir_map.keys()):
        sections.append(f"# 目录: {dir_key}")
        sections.append("文件清单:")
        for fpath in dir_map[dir_key]:
            sections.append(f"- {os.path.basename(fpath)}")
        sections.append("")

        for fpath in dir_map[dir_key]:
            rel_file = os.path.relpath(fpath, root).replace("\\", "/")
            raw = _read_text_file(fpath)
            numbered = _with_line_numbers(raw)
            sections.append(f"## 文件: {rel_file}")
            sections.append(f"```text\n{numbered}\n```")
            sections.append("")

    return "\n".join(sections).rstrip()
