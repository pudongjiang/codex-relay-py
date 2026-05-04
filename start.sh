#!/bin/zsh

# 进入脚本所在文件夹（自动识别）
cd "$(dirname "$0")" || exit

# 运行 codex-relay（你原来的命令，直接封装）
echo "正在启动 codex-relay..."
PYTHONPATH=. .venv/bin/python -m py

# 结束后暂停（防止窗口一闪而过）
echo ""
echo "按回车键退出"
read