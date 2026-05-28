import numpy as np
import os
import mmapy
from ansys.mapdl.core import launch_mapdl
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix

# =============================================================================
# 1. 全局参数
# =============================================================================
VOL_FRAC = 0.3
PENALTY = 3
MAX_ITER = 100
CONV_CRITERIA = 1e-3
RADIUS_FILTER = 2.0  

# 3D 设计域规模
XL, YL, ZL = 30, 90, 60
EDIS = 1
Ef = 1.0          
Ec = [2, 2]       
R1, R2 = 10, 5
del_ = [2, 2]     
Cnum = 2          

# =============================================================================
# 2. 启动 MAPDL 并进行高性能求解设置
# =============================================================================
# 分配 4000MB 内存，根据机器性能可调整核心数 nproc
mapdl = launch_mapdl(run_location=os.getcwd(), nproc=4, override=True, additional_switches="-m 4000 -db 2000")
mapdl.clear()

# =============================================================================
# 3. 建立 3D 模型 + 网格 + 边界
# =============================================================================
mapdl.prep7()
mapdl.et(1, 185)
mapdl.block(0, XL, 0, YL, 0, ZL)
mapdl.aesize("ALL", EDIS)
mapdl.vmesh(1)

Enum = mapdl.mesh.n_elem
NumDesVar = Enum + 3 * Cnum
print(f"单元数: {Enum}, 总设计变量: {NumDesVar}")

# 边界条件：底面固定
mapdl.nsel("S", "LOC", "Y", 0)
mapdl.d("ALL", "ALL", 0)

# 载荷条件：上表面中间
mapdl.nsel("S", "LOC", "X", XL/2-2, XL/2+2)
mapdl.nsel("R", "LOC", "Y", YL)
mapdl.nsel("R", "LOC", "Z", 0, 1)
mapdl.f("ALL", "FZ", -0.1)
mapdl.allsel()

# 高效提取单元中心坐标
mapdl.run("*dim,cent,array,%d,3" % Enum)
mapdl.run("*vget,cent(1,1),elem,1,cent,x")
mapdl.run("*vget,cent(1,2),elem,1,cent,y")
mapdl.run("*vget,cent(1,3),elem,1,cent,z")
centers = np.array(mapdl.parameters["CENT"])

# 基于 cKDTree 初始化稀疏过滤矩阵
tree = cKDTree(centers)
neighbors = tree.query_ball_point(centers, r=RADIUS_FILTER)
rows, cols, vals = [], [], []
for i, ids in enumerate(neighbors):
    d = np.linalg.norm(centers[ids] - centers[i], axis=1)
    w = np.maximum(0, RADIUS_FILTER - d)
    rows.extend([i]*len(ids))
    cols.extend(ids)
    vals.extend(w)
H = coo_matrix((vals, (rows, cols)), shape=(Enum, Enum)).tocsr()
Hs = np.array(H.sum(axis=1)).flatten()

# =============================================================================
# 4. GCMMA 初始化
# =============================================================================
n = NumDesVar
m = 2  # 约束数更新为 2（约束1：体积份额；约束2：组件干涉）

eeen = np.ones((n, 1))
eeem = np.ones((m, 1))
zeron = np.zeros((n, 1))
zerom = np.zeros((m, 1))

xval = np.ones((n, 1)) * VOL_FRAC
xval[Enum:, 0] = [15.0, 30, 30, 15, 60, 30]  # 组件初始位置

xold1 = xval.copy()
xold2 = xval.copy()

xmin = np.zeros_like(xval)
xmax = np.ones_like(xval)
xmin[:Enum] = 0.001
xmin[Enum:,0] = [10,10,10,10,10,10]
xmax[:Enum] = 1.0
xmax[Enum:,0] = [20,80,50,20,80,50]

low = xmin.copy()
upp = xmax.copy()

epsimin = 1e-6
c = 1000 * eeem
d = eeem.copy()
a0 = 1
a = zerom.copy()
raa0, raa = 0.01, 0.01 * eeem
raa0eps, raaeps = 1e-6, 1e-6 * eeem

# =============================================================================
# 5. 三场密度法核心算子（矩阵乘法提速）
# =============================================================================
def forward_filter(rho_design):
    rho_filt = H @ rho_design
    rho_filt /= Hs
    return rho_filt

def heaviside_projection(rho_filt, beta=1, eta=0.5):
    tb = np.tanh(beta * eta)
    v = np.tanh(beta * (rho_filt - eta))
    rho_phys = (tb + v) / (tb + np.tanh(beta*(1-eta)))
    dfd = (beta * (1 - v**2)) / (tb + np.tanh(beta*(1-eta)))
    return rho_phys, dfd

def adjoint_filter(dfd):
    out = H.T @ (dfd / Hs)
    return out

# =============================================================================
# 6. 联合求解函数
# =============================================================================
def solve_three_field(xval, iter_num):
    # 只取拓扑变量做三场法
    rho_design = xval[:Enum, 0].copy()
    rho_filt = forward_filter(rho_design)
    rho_phys, dfd = heaviside_projection(rho_filt)

    # 组件位置直接读取
    com1x, com1y, com1z = xval[Enum+0, 0], xval[Enum+1, 0], xval[Enum+2, 0]
    com2x, com2y, com2z = xval[Enum+3, 0], xval[Enum+4, 0], xval[Enum+5, 0]

    cx, cy, cz = centers[:,0], centers[:,1], centers[:,2]
    E = np.zeros(Enum)
    color = np.zeros(Enum)

    # ---------------- 组件1 R-function ----------------
    nx, ny, nz = cx - com1x, cy - com1y, cz - com1z
    ft1 = (R1**2 - nx**2 - ny**2 - nz**2) / del_[0]
    dft1_x, dft1_y, dft1_z = 2 * nx / del_[0], 2 * ny / del_[0], 2 * nz / del_[0]
    fA1 = np.arctan(ft1) / np.pi + 0.5
    dfA1 = 1.0 / (np.pi * (1.0 + ft1**2))

    # ---------------- 组件2 R-function ----------------
    nx, ny, nz = cx - com2x, cy - com2y, cz - com2z
    ft2 = (R2**2 - nx**2 - ny**2 - nz**2) / del_[1]
    dft2_x, dft2_y, dft2_z = 2 * nx / del_[1], 2 * ny / del_[1], 2 * nz / del_[1]
    fA2 = np.arctan(ft2) / np.pi + 0.5
    dfA2 = 1.0 / (np.pi * (1.0 + ft2**2))

    # ---------------- 物理密度进入有限元材料映射 ----------------
    E_base = Ef * (rho_phys ** PENALTY)
    E[:] = E_base
    vol = np.sum(rho_phys)
    color[:] = rho_design

    E += fA1 * (Ec[0] - E_base)
    E += fA2 * (Ec[1] - E_base)
    vol += np.sum(fA1 * (1 - rho_phys))
    vol += np.sum(fA2 * (1 - rho_phys))
    
    color += fA1 * (1.4 - rho_phys)
    color += fA2 * (1.4 - rho_phys)

# ---------------- 高效分箱材料赋值（修复版） ----------------
# ---------------- 高效分箱材料赋值（万无一失稳健版） ----------------
    mapdl.prep7()
    NBIN = 100
    Emin = 1e-6
    Emax = np.max(E)
    Ebins = np.linspace(Emin, Emax, NBIN)
    
    # 仅在首代创建离散材料库
    if iter_num == 1:
        for i in range(NBIN):
            mapdl.mp("EX", i + 1, float(Ebins[i]))
            mapdl.mp("PRXY", i + 1, 0.3)

    mat_id = np.clip(np.digitize(E, Ebins), 1, NBIN)

    # 预先清理可能残余的数组
    mapdl.run("*DEL,E_GRP,,NOPR")
    mapdl.run("ESEL,NONE")  # 先清空当前单元选择

    for m_idx in range(1, NBIN + 1):
        elems = np.where(mat_id == m_idx)[0] + 1
        if elems.size > 0:
            # 批量注入当前分箱的单元编号
            mapdl.parameters["E_GRP"] = elems.astype(np.float64)
            
            # 【核心修复】：
            # 避开不稳定的 *VPUT 标签，改用 APDL 内部高效执行的 *DO 循环批量执行选择。
            # 虽然包含 *DO，但它是在 ANSYS 内存中纯底层执行，没有任何 PyMAPDL 的网络/通信延迟！
            cmd = f"""
            ESEL,NONE
            *DO,i,1,{elems.size}
              ESEL,A,ELEM,,E_GRP(i)
            *ENDDO
            EMODIF,ALL,MAT,{m_idx}
            """
            mapdl.input_strings(cmd)
            
            # 清理临时变量
            mapdl.run("*DEL,E_GRP,,NOPR")
            
    mapdl.allsel()  # 修改完毕后恢复全选

    # 抑制警告与不必要的屏幕输出，优化磁盘 I/O
    mapdl.slashsolu()
    mapdl.run("/NERR,,,-1")
    mapdl.run("/OUTPUT,SCRATCH,TXT")
    mapdl.antype(0)
    mapdl.solve()
    mapdl.finish()

    # ---------------- 后处理数据批量读取与变量注入 ----------------
    mapdl.post1()
    mapdl.set("LAST")
    mapdl.etable("SE", "SENE")
    mapdl.etable("TOP", "TOPO")
    
    # 向量化读取应变能
    mapdl.run(f"*DIM,SEARR,ARRAY,{Enum}")
    mapdl.run("*VGET,SEARR(1),ELEM,1,ETAB,SE")
    se = np.array(mapdl.parameters["SEARR"]).ravel()
    compliance = np.sum(se)
    
    # 向量化注入拓扑变量到单元表，移除原先的 APDL *DO 循环
    mapdl.parameters["TOPARR"] = color.astype(np.float64)
    mapdl.run("*VPUT,TOPARR(1),ELEM,1,ETAB,TOP")

    # 计算组件的空间干涉重叠体积
    V_inter = np.sum(fA1 * fA2)

    return compliance, vol, se, rho_design, rho_filt, rho_phys, dfd, \
           fA1, dfA1, fA2, dfA2, dft1_x, dft1_y, dft1_z, dft2_x, dft2_y, dft2_z, V_inter

# =============================================================================
# 7. 三场法灵敏度（包含组件解析导数）
# =============================================================================
def compute_sens_three_field(se, rho_phys, dfd_proj,
                             fA1, dfA1, fA2, dfA2,
                             dft1_x, dft1_y, dft1_z,
                             dft2_x, dft2_y, dft2_z):

    E_phys = Ef * rho_phys ** PENALTY
    dC_dp = -PENALTY * rho_phys**(PENALTY-1) * Ef * se / (E_phys + 1e-9)
    dV_dp = np.ones_like(dC_dp)

    # 拓扑灵敏度 × 投影导数
    dC_df = dC_dp * dfd_proj
    dV_df = dV_dp * dfd_proj

    def comp(k, dfA, dft_x, dft_y, dft_z):
        dE = (Ec[k] - E_phys)
        s = se / (E_phys + 1e-9)
        cx = np.sum(-dfA * dft_x * dE * s)
        cy = np.sum(-dfA * dft_y * dE * s)
        cz = np.sum(-dfA * dft_z * dE * s)
        vx = np.sum(dfA * dft_x * (1 - rho_phys))
        vy = np.sum(dfA * dft_y * (1 - rho_phys))
        vz = np.sum(dfA * dft_z * (1 - rho_phys))
        return cx, cy, cz, vx, vy, vz

    c1x, c1y, c1z, v1x, v1y, v1z = comp(0, dfA1, dft1_x, dft1_y, dft1_z)
    c2x, c2y, c2z, v2x, v2y, v2z = comp(1, dfA2, dft2_x, dft2_y, dft2_z)

    return dC_df, dV_df, c1x, c1y, c1z, c2x, c2y, c2z, v1x, v1y, v1z, v2x, v2y, v2z

# =============================================================================
# 8. 主循环
# =============================================================================
iter_num = 0
change = 1.0

while iter_num < MAX_ITER and change > CONV_CRITERIA:
    iter_num += 1
    print(f"\n========== 迭代 {iter_num} ==========")

    # 求解与干涉计算
    C, V, se, rho_d, rho_f, rho_ph, dfd,\
    fA1, dfA1, fA2, dfA2, dx1, dy1, dz1, dx2, dy2, dz2, V_inter = solve_three_field(xval, iter_num)

    # 灵敏度计算
    dC_df, dV_df, c1x,c1y,c1z, c2x,c2y,c2z, v1x,v1y,v1z, v2x,v2y,v2z =\
        compute_sens_three_field(se, rho_ph, dfd, fA1, dfA1, fA2, dfA2,
                                 dx1, dy1, dz1, dx2, dy2, dz2)

    # 三场法稀疏反向滤波：只处理拓扑灵敏度
    een = adjoint_filter(dC_df)
    ven = adjoint_filter(dV_df)

    # 组装目标函数偏导数
    f0val = C
    df0dx = np.zeros((n,1))
    df0dx[:Enum, 0] = een  
    df0dx[Enum+0] = c1x; df0dx[Enum+1] = c1y; df0dx[Enum+2] = c1z
    df0dx[Enum+3] = c2x; df0dx[Enum+4] = c2y; df0dx[Enum+5] = c2z

    # 组装双重约束向量及灵敏度矩阵
    INTER_TOL = 1e-1  # 干涉容差设置
    fval = np.array([[V/Enum - VOL_FRAC], 
                     [V_inter/Enum - INTER_TOL]])

    dfdx = np.zeros((m, n))
    # 约束一：结构整体体积份额梯度
    dfdx[0, :Enum] = ven / Enum
    dfdx[0, Enum+0] = v1x/Enum; dfdx[0, Enum+1] = v1y/Enum; dfdx[0, Enum+2] = v1z/Enum
    dfdx[0, Enum+3] = v2x/Enum; dfdx[0, Enum+4] = v2y/Enum; dfdx[0, Enum+5] = v2z/Enum

    # 约束二：组件非重叠干涉梯度（对拓扑变量偏导为0，对组件位置全解析求导）
    dfdx[1, :Enum] = 0.0 
    dfdx[1, Enum+0] = np.sum(-dfA1 * dx1 * fA2) / Enum
    dfdx[1, Enum+1] = np.sum(-dfA1 * dy1 * fA2) / Enum
    dfdx[1, Enum+2] = np.sum(-dfA1 * dz1 * fA2) / Enum
    dfdx[1, Enum+3] = np.sum(-dfA2 * dx2 * fA1) / Enum
    dfdx[1, Enum+4] = np.sum(-dfA2 * dy2 * fA1) / Enum
    dfdx[1, Enum+5] = np.sum(-dfA2 * dz2 * fA1) / Enum

    # GCMMA 渐近线更新与子问题求解
    low, upp, raa0, raa = mmapy.asymp(
        iter_num, n, xval, xold1, xold2, xmin, xmax,
        low, upp, raa0, raa, raa0eps, raaeps, df0dx.ravel(), dfdx
    )
    xmma, *rest = mmapy.gcmmasub(
        m, n, iter_num, epsimin, xval, xmin, xmax, low, upp,
        raa0, raa, f0val, df0dx, fval, dfdx, a0, a, c, d
    )

    # 状态缓存更新
    xold2 = xold1.copy()
    xold1 = xval.copy()
    xval = xmma.copy()

    change = np.max(np.abs(xval - xold1))

    # 降低磁盘 I/O：每 5 代或最终收敛时输出一次拓扑 JPEG 图像
    if iter_num % 5 == 0 or change <= CONV_CRITERIA:
        mapdl.run("/view, 1, -1, 0.5, 0.25")
        mapdl.run("ESEL, S, ETAB, TOP, 0.5, 100.0")
        mapdl.run("/show,jpeg,,0")
        mapdl.pletab("TOP","NOAV")
        mapdl.run("JPEG,QUAL,100")
        mapdl.run("JPEG,COLOR,2")
        mapdl.run("/GFILE,800")
        mapdl.run("/show,close")

    print(f"柔度: {C:.4f} | 体积: {V/Enum:.3f} | 干涉体积: {V_inter/Enum:.4f} | 变化: {change:.4f}")

# =============================================================================
# 结束与数据保存
# =============================================================================
mapdl.save("TOPO_RESULT_THREE_FIELD")
mapdl.exit(force=True)
print("\n✅ 三场密度法优化完成！仅拓扑过滤，具备组件防重叠干涉约束！")
