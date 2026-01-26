from ase import units
from ase.io import read, write, Trajectory
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from nequip.ase import NequIPCalculator
import numpy as np

# ----------------------------
# 1. 加载结构和力场
# ----------------------------
atoms = read("model.xyz")

ALLOWED_ELEMENTS = ['Al', 'Mg', 'Ti', 'V', 'Cr', 'Fe', 'Co', 'Ni', 'Cu', 'Zr']

calc = NequIPCalculator.from_compiled_model(
    compile_path="../best-cpu.nequip.pth",
    device="cpu",
    species_to_type_name={el: el for el in ALLOWED_ELEMENTS}
)
atoms.calc = calc

# ----------------------------
# 2. 形状惩罚函数
# ----------------------------
def shape_penalty(cell):
    off_diag_norm = np.sqrt(
        cell[0,1]**2 + cell[0,2]**2 +
        cell[1,0]**2 + cell[1,2]**2 +
        cell[2,0]**2 + cell[2,1]**2
    )
    a, b, c = np.linalg.norm(cell[0]), np.linalg.norm(cell[1]), np.linalg.norm(cell[2])
    mean_len = (a + b + c) / 3.0
    iso_penalty = np.sqrt(((a-mean_len)**2 + (b-mean_len)**2 + (c-mean_len)**2) / 3.0)
    return off_diag_norm + iso_penalty

# ----------------------------
# 3. 晶胞扰动函数
# ----------------------------
def propose_new_cell(old_cell, volume_scale, max_shear=0.1):
    scale = volume_scale ** (1/3)
    new_cell = old_cell * scale
    shear = (np.random.rand(3,3) - 0.5) * max_shear
    np.fill_diagonal(shear, 0.0)
    new_cell = new_cell + shear @ new_cell
    return new_cell

def apply_cell(atoms, new_cell):
    frac_coords = atoms.get_scaled_positions()
    new_atoms = atoms.copy()
    new_atoms.set_cell(new_cell, scale_atoms=False)
    new_atoms.set_scaled_positions(frac_coords)
    return new_atoms

# ----------------------------
# 4. 打开轨迹文件
# ----------------------------
traj_main = Trajectory("compression_path.traj", "w")  # 主压缩路径
traj_md_snapshots = Trajectory("md_relax_snapshots.traj", "w")  # 每轮 NVT 后快照

# 保存初始结构
traj_main.write(atoms)
traj_md_snapshots.write(atoms)

# ----------------------------
# 5. 主压缩循环
# ----------------------------
T = 300.0
lambda_shape = 0.5
max_cycles = 1000
md_steps_per_cycle = 10
volume_scale_factor = 0.96
max_shear_strength = 0.08

print(f"初始体积: {atoms.get_volume():.2f} Å³")

for cycle in range(max_cycles):
    # --- (a) NVT 弛豫 ---
    MaxwellBoltzmannDistribution(atoms, T * units.kB)
    Stationary(atoms)
    dyn = VelocityVerlet(atoms, timestep=1.0 * units.fs)
    for step in range(md_steps_per_cycle):
        dyn.run(1)
        # 可选：每 50 步存一次 MD 轨迹（此处为简洁，只存末尾）
    # 保存本轮 NVT 结束后的结构
    traj_md_snapshots.write(atoms.copy())
    
    E_old = atoms.get_potential_energy()
    S_old = shape_penalty(atoms.get_cell())
    Eff_old = E_old + lambda_shape * S_old
    V_old = atoms.get_volume()
    
    # --- (b) 提议新晶胞 ---
    new_cell = propose_new_cell(atoms.get_cell(), volume_scale_factor, max_shear=max_shear_strength)
    atoms_trial = apply_cell(atoms, new_cell)
    atoms_trial.calc = calc
    
    E_new = atoms_trial.get_potential_energy()
    S_new = shape_penalty(new_cell)
    Eff_new = E_new + lambda_shape * S_new
    V_new = np.linalg.det(new_cell)
    
    # --- (c) 接受准则 ---
    dEff = Eff_new - Eff_old
    if dEff < 2.0:
        atoms = atoms_trial
        traj_main.write(atoms.copy())  # ✅ 关键：记录成功压缩步
        print(f"✅ Cycle {cycle+1}: V {V_old:.2f} → {V_new:.2f} Å³ | "
              f"E: {E_old:.2f} → {E_new:.2f} eV | Shape: {S_old:.3f} → {S_new:.3f}")
    else:
        print(f"❌ Cycle {cycle+1}: 拒绝 (ΔE_eff = {dEff:.3f} eV)")
        volume_scale_factor = min(volume_scale_factor * 1.02, 0.99)

# 关闭轨迹文件
traj_main.close()
traj_md_snapshots.close()

# ----------------------------
# 6. 最终优化并保存
# ----------------------------
print("✨ 最终优化...")
from ase.optimize import BFGS
from ase.constraints import UnitCellFilter

uf = UnitCellFilter(atoms)
opt = BFGS(uf, logfile="final_opt.log")
opt.run(fmax=0.03, steps=400)

# 保存最终结构
write("final_compact_shape_regularized.xyz", atoms)
write("final_compact_shape_regularized.cif", atoms)

# 可选：也将最终结构追加到主轨迹
with Trajectory("compression_path.traj", "a") as traj:
    traj.write(atoms)

print(f"🎉 完成！最终体积: {atoms.get_volume():.2f} Å³")
print(f"轨迹已保存至: compression_path.traj (主路径), md_relax_snapshots.traj (每轮快照)")