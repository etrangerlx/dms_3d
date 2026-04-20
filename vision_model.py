import matplotlib
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from dataclasses import dataclass

# ================= 1. 智能后端选择 =================
BACKEND_NAME = 'Agg'
try:
    import PyQt5
    matplotlib.use('Qt5Agg', force=True)
    BACKEND_NAME = 'Qt5Agg'
except ImportError:
    try:
        import tkinter
        matplotlib.use('TkAgg', force=True)
        BACKEND_NAME = 'TkAgg'
    except ImportError:
        matplotlib.use('Agg', force=True)
print(f"🖥️  当前后端: {BACKEND_NAME}")

# ================= 2. 自动配置中文字体 =================
import matplotlib.font_manager as fm

def setup_chinese_font():
    # 按优先级列出常见的中文字体名称
    candidates = [
        'Noto Sans CJK SC',   # Linux 首选 (Not CJK)
        'WenQuanYi Micro Hei', # Linux 备选 (文泉驿)
        'SimHei',             # Windows/Linux 通用 (黑体)
        'Microsoft YaHei'     # Windows 首选 (微软雅黑)
    ]
    
    # 获取系统已安装的所有字体名称
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    found_font = None
    for font in candidates:
        if font in available_fonts:
            found_font = font
            break
            
    if found_font:
        plt.rcParams['font.sans-serif'] = [found_font]
        print(f"✅ 成功加载中文字体: {found_font}")
    else:
        print("❌ 未找到中文字体，中文将显示为方块。")
        print("💡 请运行: sudo apt install fonts-noto-cjk && rm -rf ~/.cache/matplotlib")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] #  fallback
        
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

setup_chinese_font() # 执行配置

# ================= 3. 座舱部件坐标定义 =================
@dataclass
class DriverPosition:
    x: float = -400.0
    y: float = 700.0
    z: float = 300.0

def generate_windshield_points(n_width=25, n_height=20):
    """生成挡风玻璃密集点阵"""
    u = np.linspace(0, 1, n_width)
    v = np.linspace(0, 1, n_height)
    U, V = np.meshgrid(u, v)
    TL, TR, BR, BL = [-620, 900, 850], [620, 900, 850], [720, 1260, 220], [-720, 1260, 220]
    X = (1-U)*(1-V)*TL[0] + U*(1-V)*TR[0] + U*V*BR[0] + (1-U)*V*BL[0]
    Y = (1-U)*(1-V)*TL[1] + U*(1-V)*TR[1] + U*V*BR[1] + (1-U)*V*BL[1]
    Z = (1-U)*(1-V)*TL[2] + U*(1-V)*TR[2] + U*V*BR[2] + (1-U)*V*BL[2]
    # Z=np.sqrt(1-X**2-Y**2)
    Y += 40 * (1 - (2*U - 1)**2) * (1 - (2*V - 1)**2)  # 外凸曲率
    # return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    return [[x,y,z] for x,y,z in zip(X.reshape(-1),Y.reshape(-1),Z.reshape(-1))]

def generate_side_window_points(side='left', n_h=150, n_v=120):
    """生成侧窗密集点阵（梯形模拟）"""
    u = np.linspace(0, 1, n_h)
    v = np.linspace(0, 1, n_v)
    U, V = np.meshgrid(u, v)
    
    if side == 'left':
        LT, LB = [-620, 900, 850], [-720, 1260, 220]
        LTH, LBH = [-620, 700, 850], [-720, 700, 220]
        X = (1-U)*(1-V)*LT[0] + U*(1-V)*LTH[0] + U*V*LBH[0] + (1-U)*V*LB[0]
        Y = (1-U)*(1-V)*LT[1] + U*(1-V)*LTH[1] + U*V*LBH[1] + (1-U)*V*LB[1]
        Z = (1-U)*(1-V)*LT[2] + U*(1-V)*LTH[2] + U*V*LBH[2] + (1-U)*V*LB[2]

    else:
        RT, RB = [620, 900, 850], [720, 1260, 220]
        RTH, RBH = [620, 700, 850], [720, 700, 220]
        X = (1-U)*(1-V)*RT[0] + U*(1-V)*RTH[0] + U*V*RBH[0] + (1-U)*V*RB[0]
        Y = (1-U)*(1-V)*RT[1] + U*(1-V)*RTH[1] + U*V*RBH[1] + (1-U)*V*RB[1]
        Z = (1-U)*(1-V)*RT[2] + U*(1-V)*RTH[2] + U*V*RBH[2] + (1-U)*V*RB[2]
        
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

def generate_mirror_points(center, width, height, n=5):
    """生成后视镜密集点阵"""
    u = np.linspace(-width/2, width/2, n)
    v = np.linspace(-height/2, height/2, n)
    U, V = np.meshgrid(u, v)
    X = center[0] + U * 0.2
    Y = center[1] + U * 0.8
    Z = center[2] + V
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

# 生成所有密集点
WINDSHIELD_DENSE = generate_windshield_points()
WINDSHIELD_BOUNDARY = np.array([WINDSHIELD_DENSE[0], WINDSHIELD_DENSE[24], WINDSHIELD_DENSE[-1], WINDSHIELD_DENSE[-25]])
WINDSHIELD_BOUNDARY_CLOSED = np.vstack([WINDSHIELD_BOUNDARY, WINDSHIELD_BOUNDARY[0]])

LEFT_WIN_DENSE = generate_side_window_points('left')
RIGHT_WIN_DENSE = generate_side_window_points('right')

LW_B = np.array([LEFT_WIN_DENSE[0], LEFT_WIN_DENSE[149], LEFT_WIN_DENSE[-1], LEFT_WIN_DENSE[-150]])
LW_B_CLOSED = np.vstack([LW_B, LW_B[0]])
RW_B = np.array([RIGHT_WIN_DENSE[0], RIGHT_WIN_DENSE[149], RIGHT_WIN_DENSE[-1], RIGHT_WIN_DENSE[-150]])
RW_B_CLOSED = np.vstack([RW_B, RW_B[0]])

LM_CENTER = np.array([-980, 820, 430])
RM_CENTER = np.array([980, 820, 430])
LEFT_MIRROR_DENSE = generate_mirror_points(LM_CENTER, width=120, height=80)
RIGHT_MIRROR_DENSE = generate_mirror_points(RM_CENTER, width=120, height=80)

LM_B = np.array([[-980, 820, 470], [-900, 850, 470], [-900, 850, 390], [-980, 820, 390]])
LM_B_CLOSED = np.vstack([LM_B, LM_B[0]])
RM_B = np.array([[900, 850, 470], [980, 820, 470], [980, 820, 390], [900, 850, 390]])
RM_B_CLOSED = np.vstack([RM_B, RM_B[0]])

# ================= 4. 交互可视化核心类 =================
class InteractiveFOVViewer:
    def __init__(self):
        self.driver = DriverPosition()
        self.fig = plt.figure(figsize=(14, 8), facecolor='#1a1a2e')
        self.ax = self.fig.add_subplot(121, projection='3d', facecolor='#16213e')
        self.ax2 = self.fig.add_subplot(122, facecolor='#16213e')
        plt.subplots_adjust(bottom=0.18, left=0.08, right=0.95, top=0.92, wspace=0.25)
        self.fov_lines = []
        self._setup_plot()
        self._setup_sliders()

    def _setup_plot(self):
        self.ax.set_title('全车玻璃密集点阵包络', color='white', fontsize=14, pad=20)
        self.ax.set_xlabel('X 横向 (mm)', color='#aaa')
        self.ax.set_ylabel('Y 纵向 (mm)', color='#aaa')
        self.ax.set_zlabel('Z 垂向 (mm)', color='#aaa')
        self.ax.set_xlim(-1000, 1000)
        self.ax.set_ylim(200, 1400)
        self.ax.set_zlim(0, 1000)

        # 3D 渲染
        self.ax.plot_wireframe(np.array([[float(item[0])] for item in WINDSHIELD_DENSE]).reshape(20, -1),
                                np.array([[float(item[1])] for item in WINDSHIELD_DENSE]).reshape(20, -1),
                                np.array([[float(item[2])] for item in WINDSHIELD_DENSE]).reshape(20, -1), 
                                color= 'blue', alpha=0.6, linewidth=1.2)
        self.ax.plot(WINDSHIELD_BOUNDARY_CLOSED[:,0], WINDSHIELD_BOUNDARY_CLOSED[:,1], WINDSHIELD_BOUNDARY_CLOSED[:,2], 'c-', linewidth=2, label='挡风玻璃')
        # self.ax.add_collection3d(Poly3DCollection([list(zip(WINDSHIELD_BOUNDARY[:,0], WINDSHIELD_BOUNDARY[:,1], WINDSHIELD_BOUNDARY[:,2]))], facecolors='cyan', alpha=0.12))
        
        # self.ax.plot(LM_B_CLOSED[:,0], LM_B_CLOSED[:,1], LM_B_CLOSED[:,2], '#ff9900', linewidth=2, label='左后视镜')
        # self.ax.add_collection3d(Poly3DCollection([list(zip(LM_B[:,0], LM_B[:,1], LM_B[:,2]))], facecolors='#ff9900', alpha=0.5))
        
        # self.ax.plot(RM_B_CLOSED[:,0], RM_B_CLOSED[:,1], RM_B_CLOSED[:,2], '#32cd32', linewidth=2, label='右后视镜')
        # self.ax.add_collection3d(Poly3DCollection([list(zip(RM_B[:,0], RM_B[:,1], RM_B[:,2]))], facecolors='#32cd32', alpha=0.5))
        
        self.ax.plot(LW_B_CLOSED[:,0], LW_B_CLOSED[:,1], LW_B_CLOSED[:,2], '#8899aa', linewidth=1.5, label='左侧窗')
        self.ax.add_collection3d(Poly3DCollection([list(zip(LW_B[:,0], LW_B[:,1], LW_B[:,2]))], facecolors='#8899aa', alpha=0.15))
        
        self.ax.plot(RW_B_CLOSED[:,0], RW_B_CLOSED[:,1], RW_B_CLOSED[:,2], '#8899aa', linewidth=1.5, label='右侧窗')
        self.ax.add_collection3d(Poly3DCollection([list(zip(RW_B[:,0], RW_B[:,1], RW_B[:,2]))], facecolors='#8899aa', alpha=0.15))

        for i, (color, label) in enumerate(zip(['r','g','b'], ['X','Y','Z'])):
            self.ax.plot([0, 300*(i==0)], [0, 300*(i==1)], [0, 300*(i==2)], color=color, linewidth=2, label=f'{label}轴')

        self.driver_point, = self.ax.plot([], [], [], 'wo', markersize=8, label='眉心')
        self.ax.legend(fontsize=8, facecolor='#0f3460', labelcolor='white', loc='upper left')
        self.ax.grid(True, alpha=0.3, color='#444')
        self.ax.set_box_aspect([1, 1.2, 0.6])

        self.ax2.set_title('全视野角度包络 (方位角-俯仰角)', color='white', fontsize=12, pad=15)
        self.ax2.set_xlabel('方位角 φ (°)', color='#aaa')
        self.ax2.set_ylabel('俯仰角 θ (°)', color='#aaa')
        self.ax2.set_xlim(-160, 160)
        self.ax2.set_ylim(-50, 45)
        self.ax2.grid(True, alpha=0.3, color='#444', linestyle=':')
        self.ax2.axhline(0, color='#aaa', linewidth=0.8)
        self.ax2.axvline(0, color='#aaa', linewidth=0.8)

    def _setup_sliders(self):
        slider_color = '#e94560'

        ax_x = plt.axes([0.1, 0.13, 0.25, 0.04], facecolor='#0f3460')
        self.slider_x = Slider(ax_x, '左右 X (mm)', -500, -300, valinit=-400, color=slider_color)
        self.slider_x.on_changed(self._update)


        ax_y = plt.axes([0.1, 0.08, 0.25, 0.04], facecolor='#0f3460')
        self.slider_y = Slider(ax_y, '前后 Y (mm)', 500, 900, valinit=700, color=slider_color)
        self.slider_y.on_changed(self._update)

        ax_z = plt.axes([0.1, 0.03, 0.25, 0.04], facecolor='#0f3460')
        self.slider_z = Slider(ax_z, '高低 Z (mm)', 220, 800, valinit=300, color=slider_color)
        self.slider_z.on_changed(self._update)

        ax_reset = plt.axes([0.4, 0.04, 0.1, 0.05], facecolor='#e94560')
        self.btn_reset = Button(ax_reset, '重置', color='#c0392b', hovercolor='#e74c3c')
        self.btn_reset.on_clicked(self._reset)

        self.info_text = self.fig.text(0.52, 0.04, '', color='white', fontsize=8, bbox=dict(facecolor='#0f3460', edgecolor='#e94560', boxstyle='round,pad=0.4'))

    def _calc_angles(self, points):
        origin = np.array([self.driver.x, self.driver.y, self.driver.z])
        rel = points - origin
        r = np.linalg.norm(rel, axis=1)
        az = np.arctan2(rel[:, 0], rel[:, 1]) * 180/np.pi
        el = np.arctan2(rel[:, 2], np.sqrt(rel[:,0]**2 + rel[:,1]**2)) * 180/np.pi
        return r, az, el

    def _update(self, val=None):
        self.driver.x = self.slider_x.val
        self.driver.y = self.slider_y.val
        self.driver.z = self.slider_z.val
        self.driver_point.set_data([self.driver.x], [self.driver.y])
        self.driver_point.set_3d_properties([self.driver.z])

        for line in self.fov_lines: line.remove()
        self.fov_lines.clear()
        origin = np.array([self.driver.x, self.driver.y, self.driver.z])

        boundary_pts = np.vstack([WINDSHIELD_BOUNDARY, 
                                #   LM_B, RM_B, 
                                  LW_B, RW_B
                                  ])
        for pt in boundary_pts:
            self.fov_lines.append(self.ax.plot([origin[0], pt[0]], [origin[1], pt[1]], [origin[2], pt[2]], color='#ffff00', ls='--', lw=0.6, alpha=0.3)[0])

        # 计算角度
        _, az_ws, el_ws = self._calc_angles(WINDSHIELD_DENSE)
        _, az_lm, el_lm = self._calc_angles(LEFT_MIRROR_DENSE)
        _, az_rm, el_rm = self._calc_angles(RIGHT_MIRROR_DENSE)
        _, az_lw, el_lw = self._calc_angles(LEFT_WIN_DENSE)
        _, az_rw, el_rw = self._calc_angles(RIGHT_WIN_DENSE)

        # 更新 2D 包络图
        self.ax2.clear()
        self.ax2.set_title('全视野角度包络 (方位角-俯仰角)', color='white', fontsize=12, pad=15)
        self.ax2.set_xlabel('方位角 φ (°)', color='#aaa')
        self.ax2.set_ylabel('俯仰角 θ (°)', color='#aaa')
        self.ax2.set_xlim(-160, 160)
        self.ax2.set_ylim(-90, 90)
        self.ax2.grid(True, alpha=0.3, color='#444', linestyle=':')
        self.ax2.axhline(0, color='#aaa', linewidth=0.8)
        self.ax2.axvline(0, color='#aaa', linewidth=0.8)

        # 凸包计算
        def plot_hull(az, el, col, lab, alp=0.4):
            points = np.column_stack([az, el])
            if len(points) > 3:
                try:
                    hull = ConvexHull(points)
                    idx = hull.vertices
                    az_c = np.concatenate([az[idx], [az[idx[0]]]])
                    el_c = np.concatenate([el[idx], [el[idx[0]]]])
                    if os.path.exists("convex.txt"):
                        os.remove("convex.txt")
                    with open("convex.txt","w") as of:
                        for az_,el_ in zip(az_c,el_c):
                            of.write(str(f"{float(az_)},{float(el_)}\n"))
                    # self.ax2.fill(az_c, el_c, color=col, alpha=alp, label=lab)
                    self.ax2.plot(az_c, el_c, color=col, lw=1.5)
                except: pass
        self.ax2.plot(az_ws.reshape(20, -1), el_ws.reshape(20, -1), 'b-', linewidth=1, alpha=0.6)# 垂直线
        self.ax2.plot(az_ws.reshape(20, -1).T, el_ws.reshape(20, -1).T, 'b-', linewidth=1, alpha=0.6)# 水平线
        plot_hull(az_ws, el_ws, '#e94560', '前向视野', 0.25)
        # plot_hull(az_lm, el_lm, '#ff9900', '左后视镜', 0.6)
        # plot_hull(az_rm, el_rm, '#32cd32', '右后视镜', 0.6)
        plot_hull(az_lw, el_lw, "#a03b00", '左侧窗', 0.4)
        plot_hull(az_rw, el_rw, '#a03b00', '右侧窗', 0.4)
        
        
        ######## 强标线 ########
        # lines = [Line2D([-50, -50], [90, -30]),Line2D([-50, 50], [-30, -30]),Line2D([50, 50], [-30, 90])]  # 创建垂直线
        # for line in lines:
        #     line.set_color('r')
        #     self.ax2.add_line(line)
        # line = Line2D([-50, -50], [-30, 30])  # 创建垂直线
        # self.ax2.add_line(line) 
        self.ax2.legend(fontsize=9, facecolor='#0f3460', labelcolor='white', loc='upper right')

        self.info_text.set_text(f"眉心: ({self.driver.x:+.0f}, {self.driver.y:+.0f}, {self.driver.z:+.0f}) mm\n前挡: φ[{az_ws.min():+.1f}°~{az_ws.max():+.1f}°]\n左镜: φ[{az_lm.min():+.1f}°~{az_lm.max():+.1f}°]  左窗: φ[{az_lw.min():+.1f}°~{az_lw.max():+.1f}°]")
        self.fig.canvas.draw_idle()

    def _reset(self, event):
        self.slider_x.reset()
        self.slider_y.reset()
        self.slider_z.reset()

    def show(self):
        self._update()
        if BACKEND_NAME == 'Agg':
            save_path = 'cockpit_all_dense_fixed.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=self.fig.get_facecolor())
            print(f"🖼️  已保存至: {os.path.abspath(save_path)}")
        else:
            plt.show()

if __name__ == "__main__":
    InteractiveFOVViewer().show()