"""
Jittor Windows 环境初始化脚本。
在导入 jittor 之前先 import 此模块，解决 Windows 上的 DLL 和环境变量问题。

使用方法:
    import setup_jittor_env  # 必须在 import jittor 之前
    import jittor as jt
    
或者直接运行项目:
    python -c "import setup_jittor_env; exec(open('main.py').read())"
"""
import os
import sys

def setup():
    # 1. 清除可能残留的 use_data_gz 环境变量（防止跳过 data.cc 编译）
    os.environ.pop('use_data_gz', None)
    
    # 2. 添加 Jittor CUDA 11.2 的 DLL 搜索路径
    cuda_bin = os.path.join(
        os.path.expanduser('~'), '.cache', 'jittor',
        'jtcuda', 'cuda11.2_cudnn8_win', 'bin'
    )
    if os.path.isdir(cuda_bin):
        os.add_dll_directory(cuda_bin)

setup()
