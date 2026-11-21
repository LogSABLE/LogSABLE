#!/bin/bash
# Fetches the official LogFormer code used by run_logformer.py.
set -e
B=$(cd "$(dirname "$0")" && pwd)
[ -d "$B/logformer_official" ] || git clone https://github.com/HC-Guo/LogFormer.git "$B/logformer_official"
git -C "$B/logformer_official" checkout 2b3b8090c41cce6850cd08e8a2aecba13a6f033d
