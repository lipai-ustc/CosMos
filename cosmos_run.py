#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CoSMoS全局优化运行脚本

功能: 读取POSCAR结构文件和DeepMD势能模型，以盒子中心为核心区执行CoSMoS全局搜索
使用方法: python cosmos_run.py -p POSCAR -d potential.pb -o cosmos_results
"""
import os
import json
import argparse
from pathlib import Path
import numpy as np
from ase.io import read
from ase.calculators.deepmd import DeepMD
from cosmos_search import CoSMoSSearch
from cosmos_utils import load_potential


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    # Get current working directory where input files should be
    cwd = os.getcwd()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CoSMoS Global Structure Search')
    parser.add_argument('-i', '--input', default=os.path.join(cwd, 'input.json'),
                      help='Path to input configuration file (default: ./input.json)')
    parser.add_argument('-s', '--structure', default=os.path.join(cwd, 'init.xyz'),
                      help='Path to initial structure file (default: ./init.xyz)')
    args = parser.parse_args()

    # Validate input files exist
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not os.path.exists(args.structure):
        raise FileNotFoundError(f"Structure file not found: {args.structure}")

    # Load configuration and structure
    config = load_config(args.input)
    atoms = load_initial_structure(args.structure)

    # 根据配置创建系统...
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='使用CoSMoS算法进行全局结构优化')
    parser.add_argument('-p', '--poscar', required=True, help='POSCAR结构文件路径')
    parser.add_argument('-d', '--deepmd', required=True, help='DeepMD势能文件路径(.pb)')
    parser.add_argument('-o', '--output', default='cosmos_results', help='输出目录')
    parser.add_argument('--steps', type=int, default=100, help='CoSMoS迭代步数')
    parser.add_argument('--ds', type=float, default=0.2, help='步长 (Å)')
    parser.add_argument('--H', type=int, default=14, help='高斯势数量')
    parser.add_argument('--radius', type=float, default=10.0, help='核心区半径 (Å)')
    parser.add_argument('--decay', type=float, default=5.0, help='衰减长度 (Å)')
    args = parser.parse_args()

    # 读取POSCAR结构
    print(f"读取结构文件: {args.poscar}")
    atoms = read(args.poscar, format='vasp')
    if not atoms:
        raise ValueError("无法读取POSCAR文件，请检查路径是否正确")

    # 计算盒子中心坐标
    cell = atoms.get_cell()
    box_center = np.mean(cell, axis=0) / 2
    print(f"盒子中心坐标: {box_center} Å")

    # 设置DeepMD计算器
    print(f"加载DeepMD势能: {args.deepmd}")
    dp_calculator = DeepMD(potential=args.deepmd)

    # 初始化SSW搜索，以盒子核心为核心区
    print("初始化CoSMoS搜索...")
    ssw = CoSMoSSearch(
        initial_atoms=atoms,
        calculator=dp_calculator,
        ds=args.ds,
        output_dir=args.output,
        H=args.H,
        temperature=300,
        mobility_control=True,
        control_type='sphere',
        control_center=box_center,
        control_radius=args.radius,
        decay_type='gaussian',
        decay_length=args.decay
    )

    # 运行SSW全局优化
    print(f"开始CoSMoS全局搜索，共{args.steps}步...")
    ssw.run(steps=args.steps)
    
    # 获取结果并输出摘要
    minima_pool = ssw.get_minima_pool()
    energies = [minima.get_potential_energy() for minima in minima_pool]

    print("\nCoSMoS搜索完成!")
    print(f"找到{len(minima_pool)}个能量极小值结构")
    print(f"最低能量: {min(energies):.6f} eV")
    print(f"结果已保存至: {args.output}")


if __name__ == '__main__':
    main()
