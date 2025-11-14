#!/bin/bash

# 安装必要的Python依赖包
pip install numpy>=1.21.0
pip install ase>=3.22.1
pip install dscribe>=1.2.0

# 验证安装
echo "依赖包安装完成！"
python -c "import numpy, ase, dscribe; print('所有依赖包已成功导入')"
