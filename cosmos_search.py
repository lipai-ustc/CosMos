# cosmos_search.py (v2: faithful to Shang & Liu 2013)
# 实现CoSMoS(Core-Sampled Mobility Search)全局优化算法
# 参考文献1: Shang, R., & Liu, J. (2013). Stochastic surface walking method for global optimization of atomic clusters and biomolecules. The Journal of Chemical Physics, 139(24), 244104.
# 参考文献2：J. Chem. Theory Comput. 2012, 8, 2215
import os
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import write
from dscribe.descriptors import SOAP

class BiasedCalculator(Calculator):
    """
    带偏置势的计算器，用于在CoSMoS算法的爬坡阶段修改势能面(PES)
    在原始势能基础上叠加多个正的高斯型偏置势，引导结构向高能区域探索
    单个高斯势形式：V_bias = w * exp(-(d · (R - R1))^2 / (2 * σ^2))
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, base_calculator, gaussian_params, ds=0.2, control_center=None, control_radius=10.0, mobility_weights=None, wall_strength=10.0, wall_offset=2.0):
        super().__init__()
        self.base_calc = base_calculator  # 原始势能计算器
        self.gaussian_params = gaussian_params  # 高斯势参数列表，每个元素包含(d, R1, w)
        self.ds = ds  # 步长参数，用于高斯势宽度
        self.control_center = control_center  # 控制中心坐标
        self.control_radius = control_radius  # 核心区半径
        self.wall_offset = wall_offset  # 壁势能距离偏移
        self.wall_strength = wall_strength  # 壁势能强度(eV/Å²)
        self.mobility_weights = mobility_weights if mobility_weights is not None else np.ones(0)

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        
        # 获取原始能量和力
        atoms_base = atoms.copy()
        atoms_base.calc = self.base_calc
        E0 = atoms_base.get_potential_energy()
        F0 = atoms_base.get_forces().flatten()  # (3N,)

        # 当前位置
        R = atoms.positions.flatten()  # (3N,)

        # 总偏置能量和力初始化为0
        V_bias_total = 0.0
        F_bias_total = np.zeros_like(R)

        # 计算壁势能防止核心区原子外溢
        wall_energy = 0.0
        wall_forces = np.zeros_like(R)
        wall_radius = self.control_radius + self.wall_offset
        n_atoms = len(atoms)

        if self.control_center is not None and len(self.mobility_weights) == n_atoms:
            for i in range(n_atoms):
                # 仅对核心区原子应用壁势能
                if self.mobility_weights[i] == 1.0:
                    # 原子位置(未flattened)
                    pos = atoms.positions[i]
                    delta_R = pos - self.control_center
                    distance = np.linalg.norm(delta_R)

                    # 当原子距离超过壁半径时添加二次排斥势能
                    if distance > wall_radius:
                        delta = distance - wall_radius
                        # 势能: V_wall = 0.5 * strength * delta²
                        wall_energy += 0.5 * self.wall_strength * (delta ** 2)

                        # 力: F = -strength * delta * (delta_R/distance)，转换为flattened索引
                        force = -self.wall_strength * delta * (delta_R / distance) if distance > 0 else 0
                        wall_forces[i*3 : (i+1)*3] = force

        # 叠加所有高斯势 - 按照论文公式(5)和(6)
        for g_param in self.gaussian_params:
            # g_param应该是一个包含(d, R1, w)的元组或列表
            if len(g_param) != 3:
                continue
                
            d, R1, w = g_param
            R1_flat = R1.flatten()
            dr = R - R1_flat
            # 计算投影: (R - R1)·Nn
            proj = np.dot(dr, d)
            
            # 论文公式(6)中的参数a
            a = 2.0  # 控制高斯势的宽度和强度
            
            # 计算单个高斯势的能量贡献 - 按照论文公式(5)
            V_bias = w * np.exp(-(proj**2) / (2 * self.ds**2))
            V_bias_total += V_bias
            
            # 计算单个高斯势的力贡献 - 根据论文公式(5)求导
            # F = -dV/dR = w * exp(...) * (proj / ds²) * d
            F_bias = w * np.exp(-(proj**2) / (2 * self.ds**2)) * (proj / self.ds**2) * d
            F_bias_total += F_bias

        # 总能量和力是原始值加上偏置值和壁势能
        self.results['energy'] = E0 + V_bias_total + wall_energy
        self.results['forces'] = (F0 + F_bias_total + wall_forces).reshape((-1, 3))


class CoSMoSSearch:
    """
    实现核心采样移动搜索(CoSMoS)全局优化算法
    Core-Sampled Mobility Search (CoSMoS) 全局优化算法，结合核心区控制和原子移动性权重
    该算法通过在势能面上添加高斯偏置势引导结构探索，结合局部优化和蒙特卡洛接受准则
    高效寻找原子结构的全局极小值和反应路径
    """
    def __init__(
        self,
        initial_atoms: Atoms,
        calculator,
        soap_species=None,
        ds=0.2,              # 步长 (Å)  也是高斯势的宽度
        duplicate_tol=0.01,
        fmax=0.05,
        max_steps=500,
        output_dir="cosmos_output",
        H=14,                # 高斯势数量
        w=0.1,               # 高斯势高度 (eV)
        temperature=300,     # 温度 (K) 用于Metropolis准则
        # 新增原子移动性控制参数
        mobility_control=False,       # 是否启用位置相关移动控制
        control_type='sphere',        # 控制类型: 'sphere' (球体) 或 'plane' (平面)
        control_center=None,          # 控制中心坐标 (球体中心或平面上一点)
        control_radius=10.0,          # 核心区半径 (Å)
        plane_normal=None,            # 平面法向量 (仅用于平面控制)
        decay_type='gaussian',        # 衰减类型: 'linear' 或 'gaussian'
        decay_length=5.0              # 衰减长度 (Å)
    ):
        # 初始化极小值池
        self.pool = []  # 存储找到的极小值结构
        self.atoms = initial_atoms.copy()
        self.base_calc = calculator  # 用于计算真实势能的计算器
        self.soap_species = soap_species or list(set(initial_atoms.get_chemical_symbols()))
        self.ds = ds  # 步长参数，控制每次结构移动的距离
        self.duplicate_tol = duplicate_tol  # 结构相似度阈值，用于判断是否为新结构
        self.fmax = fmax  # 优化收敛判据，最大力阈值(eV/Å)
        self.max_steps = max_steps  # 算法最大迭代步数
        self.output_dir = output_dir  # 输出目录
        self.H = H  # 最大高斯势数量，控制爬坡阶段的探索深度
        self.w = w  # 高斯势高度，控制偏置强度
        self.wall_strength = wall_strength  # 壁势能强度 (eV/Å²)
        self.wall_offset = wall_offset  # 壁势能距离偏移 (Å)
        self.temperature = temperature  # Metropolis准则中的温度参数
        self.k_boltzmann = 8.617333262e-5  # 玻尔兹曼常数 (eV/K)
        
        # 初始化移动性控制参数
        self.mobility_control = mobility_control
        if mobility_control:
            # 默认使用盒子中心作为控制中心
            self.control_center = control_center or np.mean(initial_atoms.get_cell(), axis=0)/2
            self.control_radius = control_radius
            self.control_type = control_type
            # 平面控制参数
            if control_type == 'plane' and plane_normal is None:
                # 默认平面法向量为z轴
                self.plane_normal = np.array([0, 0, 1])
            else:
                self.plane_normal = plane_normal
            self.decay_type = decay_type
            self.decay_length = decay_length
            # 预计算初始移动权重
            self._update_mobility_weights()
        
        os.makedirs(output_dir, exist_ok=True)
        self.pool = []  # 存储找到的所有极小值结构
        self.real_energies = []  # 存储对应结构的真实能量

        # 初始结构优化（真实势）
        self.atoms.calc = self.base_calc
        self._local_minimize(self.atoms)
        self._add_to_pool(self.atoms)

    def _local_minimize(self, atoms, calc=None, fmax=None):
        """
        使用LBFGS算法进行局部结构优化
        符合CoSMoS算法文档步骤4和6的要求，采用有限内存BFGS优化器提高效率
        
        参数:
            atoms: 待优化的原子结构
            calc: 用于优化的计算器，默认为base_calc
            fmax: 收敛力阈值，默认为类实例的fmax
        """
        if calc is not None:
            atoms.calc = calc
        # 使用LBFGS优化器替代BFGS，符合文档步骤4和6的要求
        from ase.optimize import LBFGS
        opt = LBFGS(atoms, logfile=None)
        opt.run(fmax=fmax or self.fmax)

    def _add_to_pool(self, atoms):
        """
        将优化后的结构添加到极小值池，并保存到文件
        """
        real_e = atoms.get_potential_energy()
        self.pool.append(atoms.copy())
        self.real_energies.append(real_e)
        idx = len(self.pool) - 1
        write_minima(os.path.join(self.output_dir, f"minima_{idx:04d}.xyz"), atoms, real_e)
        print(f"Found new minimum #{idx}: E = {real_e:.6f} eV")

    def _update_mobility_weights(self):
        """
        根据原子坐标更新移动权重M
        根据控制类型计算原子到核心区的距离：
        - 'sphere': 原子到控制中心的距离
        - 'plane': 原子到平面的距离（平面由control_center和plane_normal定义）
        """
        if not self.mobility_control:
            self.mobility_weights = np.ones(len(self.atoms))
            return
        
        positions = self.atoms.get_positions()
        distances = np.zeros(len(self.atoms))
        
        if self.control_type == 'sphere':
            # 计算原子到球心的距离
            distances = np.linalg.norm(positions - self.control_center, axis=1)
        elif self.control_type == 'plane':
            # 计算原子到平面的距离
            # 平面方程: normal · (x - point) = 0
            # 距离公式: |normal · (x - point)| / ||normal||
            vectors = positions - self.control_center
            normal = self.plane_normal / np.linalg.norm(self.plane_normal)
            distances = np.abs(np.dot(vectors, normal))
        
        # 计算移动权重 (0-1之间)
        self.mobility_weights = np.zeros(len(self.atoms))
        for i, dist in enumerate(distances):
            if dist <= self.control_radius:
                # 核心区内，权重为1
                self.mobility_weights[i] = 1.0
            else:
                # 核心区外，根据衰减类型计算权重
                r = dist - self.control_radius
                if self.decay_type == 'linear':
                    # 线性衰减
                    self.mobility_weights[i] = max(0, 1 - r/self.decay_length)
                elif self.decay_type == 'gaussian':
                    # 高斯衰减
                    self.mobility_weights[i] = np.exp(-(r**2)/(2*self.decay_length**2))
                else:
                    # 默认不衰减（突然截止）
                    self.mobility_weights[i] = 0.0

    def _generate_random_direction(self, atoms):
        """
        生成随机搜索方向，结合全局软移动和局部刚性移动
        符合CoSMoS算法步骤1：生成初始随机方向N⁰
        
        返回:
            N: 归一化的随机方向向量
        """
        n_atoms = len(atoms)
        # 生成全局软移动方向Ns (符合麦克斯韦-玻尔兹曼分布)
        temperature = 300  # K
        mass = 1.0  # 原子质量单位
        k_boltzmann = 8.617333262e-5  # eV/K
        scale = np.sqrt(k_boltzmann * temperature / mass)
        Ns = np.random.normal(0, scale, 3 * n_atoms)
        
        # 应用移动权重: 将权重扩展为3N维度并应用到随机方向
        if self.mobility_control:
            # 更新权重（原子位置可能已变化）
            self._update_mobility_weights()
            # 将每个原子的权重应用到其3个坐标分量上
            atom_weights = np.repeat(self.mobility_weights, 3)
            Ns = Ns * atom_weights
        
        # 生成局部刚性移动方向Nl (非邻近原子成键模式)
        Nl = np.zeros(3 * n_atoms)
        if n_atoms >= 2:
            # 随机选择两个非邻近原子
            # 筛选核心区内的原子
            # 使用mobility_weights判断核心区原子 (权重为1.0表示在核心区内)
            core_atoms = np.where(self.mobility_weights == 1.0)[0].tolist()
            
            # 检查核心区原子数量是否足够
            if len(core_atoms) < 2:
                raise ValueError("核心区原子数量不足2个，无法进行计算。")
            
            max_attempts = 50
            attempts = 0
            found = False
            
            while attempts < max_attempts and not found:
                # 从核心区内选择两个非近邻原子
                indices = np.random.choice(core_atoms, 2, replace=False)
                i, j = indices
                qi = atoms.positions[i].flatten()
                qj = atoms.positions[j].flatten()
                distance = np.linalg.norm(qi - qj)
                
                if distance > 3.0:  # 仅当原子距离大于3Å时
                    # 按照论文中的公式(2)生成局部刚性移动方向
                    Nl[3*i:3*i+3] = qj
                    Nl[3*j:3*j+3] = qi
                    norm = np.linalg.norm(Nl)
                    if norm == 0:
                        raise ValueError("Nl方向向量的模长为零，无法进行归一化。")
                    Nl /= norm
                    found = True
                
                attempts += 1
            
            if not found:
                raise ValueError(f"尝试{max_attempts}次后仍未找到距离大于3Å的原子对，无法生成局部刚性移动方向。")
        
        # 混合方向向量 - 按照论文公式(1)
        lambda_param = np.random.uniform(0.1, 1.5)
        # 确保Ns和Nl都是归一化的
        Ns_norm = np.linalg.norm(Ns)
        Nl_norm = np.linalg.norm(Nl)
        
        if Ns_norm > 0:
            Ns = Ns / Ns_norm
        if Nl_norm > 0:
            Nl = Nl / Nl_norm
            
        N = Ns + lambda_param * Nl
        N /= np.linalg.norm(N) if np.linalg.norm(N) > 0 else 1
        return N

    def _apply_random_move(self, atoms, direction):
        """应用随机移动到原子结构"""
        new_atoms = atoms.copy()
        # 应用移动权重
        if self.mobility_control:
            atom_weights = np.repeat(self.mobility_weights, 3)
            direction = direction * atom_weights
        # 应用方向向量和步长
        new_pos = new_atoms.get_positions().flatten() + self.ds * direction
        new_atoms.set_positions(new_pos.reshape(-1, 3))
        return new_atoms

    def _get_real_energy(self, atoms):
        """
        获取结构在真实势能面上的能量
        """
        temp_atoms = atoms.copy()
        temp_atoms.calc = self.base_calc
        return temp_atoms.get_potential_energy()
    
    def _write_step_output(self, step, atoms, energy):
        """
        输出每一步的信息到文件
        """
        with open(os.path.join(self.output_dir, 'cosmos_log.txt'), 'a') as f:
            f.write(f"Step {step+1}: Energy = {energy:.6f} eV\n")
    
    def run(self, steps=100):
        """
        运行CoSMoS全局搜索算法，严格按照论文中的算法步骤实现
        
        参数:
            steps: 算法总迭代步数
        """
        # 初始化日志文件
        with open(os.path.join(self.output_dir, 'ssw_log.txt'), 'w') as f:
            f.write("CoSMoS Search Log\n")
            f.write(f"Initial structure: Energy = {self._get_real_energy(self.atoms):.6f} eV\n")
        
        # 初始化当前结构为初始极小值结构
        current_atoms = self.atoms.copy()
        current_energy = self._get_real_energy(current_atoms)
        
        for step in range(steps):
            print(f"\n--- CoSMoS Step {step + 1}/{steps} ---")
            
            # 算法步骤1: 在当前最小值Rm生成初始随机方向N⁰
            direction = self._generate_random_direction(current_atoms)
            
            # 算法步骤2: 使用偏置dimer旋转方法优化方向，得到N¹
            optimized_direction = self._biased_dimer_rotation(current_atoms, direction)
            
            # 算法步骤3-4: 爬坡阶段 - 添加高斯势并局部优化
            climb_atoms = current_atoms.copy()
            gaussian_params = []
            Emax = current_energy  # 记录爬坡过程中的最高能量
            
            for n in range(1, self.H + 1):
                # 计算方向向量
                if n == 1:
                    # 第一步使用优化后的方向
                    N = optimized_direction
                else:
                    # 后续步骤生成新的随机方向
                    new_rand_dir = self._generate_random_direction(climb_atoms)
                    # 使用偏置dimer旋转优化新方向
                    N = self._biased_dimer_rotation(climb_atoms, new_rand_dir)
                
                # 应用移动权重
                if self.mobility_control:
                    self._update_mobility_weights()
                    atom_weights = np.repeat(self.mobility_weights, 3)
                    N = N * atom_weights
                    N /= np.linalg.norm(N) if np.linalg.norm(N) > 0 else 1
                
                # 添加新的高斯势
                g_param = self._add_gaussian(climb_atoms, N)
                gaussian_params.append(g_param)
                
                # 创建带偏置势的计算器
                biased_calc = BiasedCalculator(
                    base_calculator=self.base_calc,
                    gaussian_params=gaussian_params,
                    ds=self.ds,
                    control_center=self.control_center,
                    control_radius=self.control_radius,
                    wall_strength=self.wall_strength,
                    wall_offset=self.wall_offset,
                    mobility_weights=self.mobility_weights
                )
                
                # 在修改后的势能面上局部优化
                climb_atoms.calc = biased_calc
                self._local_minimize(climb_atoms)
                
                # 计算真实能量
                current_climb_energy = self._get_real_energy(climb_atoms)
                Emax = max(Emax, current_climb_energy)
                
                # 算法步骤5: 检查是否满足停止条件
                if n >= self.H or current_climb_energy < current_energy:
                    print(f"爬坡阶段结束: n={n}, 当前能量 {current_climb_energy:.6f} eV")
                    break
            
            # 算法步骤6: 移除所有偏置势并在真实势能面上优化
            climb_atoms.calc = self.base_calc
            self._local_minimize(climb_atoms)
            relaxed_energy = self._get_real_energy(climb_atoms)
            
            # 算法步骤7: 使用Metropolis准则接受或拒绝
            delta_E = relaxed_energy - current_energy
            if delta_E > 0:
                accept_prob = np.exp(-delta_E / (self.k_boltzmann * self.temperature))
            else:
                accept_prob = 1.0
            
            if np.random.rand() < accept_prob:
                print(f"接受新结构: ΔE = {delta_E:.6f} eV, P = {accept_prob:.4f}")
                # 检查是否为新结构（非重复）
                if not is_duplicate_by_desc(climb_atoms, self.pool, self.soap_species, self.duplicate_tol):
                    current_atoms = climb_atoms.copy()
                    current_energy = relaxed_energy
                    self._add_to_pool(current_atoms)
                else:
                    print("新结构与已知结构重复，未添加到池中")
            else:
                print(f"拒绝新结构: ΔE = {delta_E:.6f} eV, P = {accept_prob:.4f}")
            
            # 输出当前步骤信息
            self._write_step_output(step, current_atoms, current_energy)
        
        print("\nCoSMoS搜索完成!")
        return self.pool, self.real_energies

    def _biased_dimer_rotation(self, atoms, initial_direction):
        """
        实现偏置dimer旋转方法，用于更新爬坡方向
        按照论文中描述的偏置dimer旋转方法，通过旋转初始方向并计算力来确定最佳爬坡方向
        
        参数:
            atoms: 当前原子结构
            initial_direction: 初始方向向量
        
        返回:
            优化后的爬坡方向向量
        """
        # 确保初始方向是单位向量
        N = initial_direction / np.linalg.norm(initial_direction) if np.linalg.norm(initial_direction) > 0 else initial_direction
        
        # 设置dimer长度（两个图像之间的距离）
        delta_R = 0.005  # 根据论文，典型值为0.005 Å
        
        # 计算R0和R1的位置
        pos_flat = atoms.positions.flatten()
        R0 = pos_flat - 0.5 * delta_R * N
        R1 = pos_flat + 0.5 * delta_R * N
        
        # 创建两个临时结构并计算力
        temp_atoms0 = atoms.copy()
        temp_atoms0.set_positions(R0.reshape(-1, 3))
        temp_atoms0.calc = self.base_calc
        F0 = temp_atoms0.get_forces().flatten()
        
        temp_atoms1 = atoms.copy()
        temp_atoms1.set_positions(R1.reshape(-1, 3))
        temp_atoms1.calc = self.base_calc
        F1 = temp_atoms1.get_forces().flatten()
        
        # 根据论文公式(4)计算曲率信息
        C = np.dot((F0 - F1), N) / delta_R
        
        # 执行旋转优化 - 寻找能量最小的方向
        best_direction = N
        best_value = float('inf')
        
        # 尝试多个旋转角度
        for angle in np.linspace(0, 2*np.pi, 24):  # 更密集的角度采样
            # 创建旋转轴 - 垂直于N的随机轴
            # 找到一个不平行于N的向量
            if abs(N[0]) < 0.9:
                axis = np.array([0, 1, 0])
            else:
                axis = np.array([1, 0, 0])
            # 确保轴与N正交
            axis = axis - np.dot(axis, N) * N
            axis /= np.linalg.norm(axis)
            
            # 应用旋转到方向向量
            rotated_N = self._rotate_vector(N, axis, angle)
            
            # 计算旋转后的向量值 - 按照论文中的优化准则
            # 这里使用力的投影作为优化目标
            test_atoms = atoms.copy()
            test_pos = pos_flat + rotated_N * delta_R
            test_atoms.set_positions(test_pos.reshape(-1, 3))
            test_atoms.calc = self.base_calc
            test_energy = test_atoms.get_potential_energy()
            # 添加文献中的二次偏置势能 V_N = -0.5 * a * proj²
            a = 1.0 / (self.ds ** 2)  # 使用ds参数计算a，保持与高斯势参数一致性
            proj = delta_R * np.dot(N, rotated_N)  # 计算投影项 (r1 - r0)·N_i^0
            V_N = -0.5 * a * (proj ** 2)  # 注意文献中要求的负号
            test_energy += V_N
            
            # 记录最佳方向
            if test_energy < best_value:
                best_value = test_energy
                best_direction = rotated_N
        
        return best_direction / np.linalg.norm(best_direction) if np.linalg.norm(best_direction) > 0 else best_direction

    def _rotate_vector(self, v, axis, angle):
        """使用罗德里格斯公式旋转向量
        参考: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        """
        v = np.asarray(v)
        axis = np.asarray(axis)
        axis = axis / np.linalg.norm(axis)
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        cross = np.cross(axis, v)
        dot = np.dot(axis, v)
        return v * cos_theta + cross * sin_theta + axis * dot * (1 - cos_theta)

    def _add_gaussian(self, atoms, direction):
        """
        生成新的高斯势参数，按照论文中的定义
        
        参数:
            atoms: 当前原子结构
            direction: 高斯势的方向向量
            
        返回:
            tuple: (d, R1, w) - 包含单位方向向量、参考位置和高度参数的元组
        """
        # 确保方向向量是单位向量
        norm = np.linalg.norm(direction)
        d = direction / norm if norm > 0 else direction
        
        # 使用当前原子位置作为参考位置R1
        R1 = atoms.positions.copy()
        
        # 根据爬坡阶段自适应调整高斯势高度
        # 早期步骤使用较小的w，后期步骤使用较大的w以鼓励进一步探索
        w = self.w  # 可以根据需要调整策略
        
        return (d, R1, w)
    
    def get_minima_pool(self):
        """
        获取找到的所有极小值结构池
        
        返回:
            list: 包含所有极小值Atoms结构的列表
        """
        return self.pool

# 辅助函数



def compute_soap_descriptor(atoms, species, rcut=5.0, nmax=6, lmax=6):
    """
    计算原子结构的SOAP(平滑重叠原子位置)描述符
    SOAP描述符用于量化结构相似性，对原子排列和化学环境敏感
    
    参数:
        atoms: ASE Atoms对象，包含原子结构信息
        species: 列表，包含系统中可能出现的元素符号
        rcut: 截断半径，控制原子环境的范围
        nmax: 径向基函数的数量
        lmax: 球谐函数的最大角量子数
    
    返回:
        numpy数组: 平均化的SOAP描述符向量
    """
    # 初始化SOAP描述符计算器
    soap = SOAP(species=species, periodic=True, rcut=rcut, nmax=nmax, lmax=lmax)
    # 计算描述符并在原子维度上平均，得到结构级描述符
    return soap.create(atoms).mean(axis=0)  # 平均描述符


def is_duplicate_by_desc(new_atoms, pool, species, tol=0.01):
    """
    通过SOAP描述符判断新结构是否为已有结构的重复
    使用描述符向量的欧氏距离作为相似性度量
    
    参数:
        new_atoms: ASE Atoms对象，待检查的新结构
        pool: 列表，包含已有的Atoms结构
        species: 列表，包含系统中可能出现的元素符号
        tol: 距离阈值，小于此值判定为重复结构
    
    返回:
        bool: 如果是重复结构返回True，否则返回False
    """
    if not pool:
        return False
    # 计算新结构的SOAP描述符
    desc_new = compute_soap_descriptor(new_atoms, species)
    # 与池中所有结构比较
    for atoms in pool:
        desc_old = compute_soap_descriptor(atoms, species)
        # 计算描述符向量的欧氏距离
        if np.linalg.norm(desc_new - desc_old) < tol:
            return True
    return False


def write_minima(filename, atoms, energy):
    """
    将极小值结构及其能量信息写入XYZ文件
    
    参数:
        filename: 字符串，输出文件路径
        atoms: ASE Atoms对象，包含极小值结构
        energy: 浮点数，结构的能量值(eV)
    """
    # 将能量信息存储在atoms对象的info字典中
    atoms.info['energy'] = energy
    # 使用ASE的write函数写入XYZ文件
    write(filename, atoms)