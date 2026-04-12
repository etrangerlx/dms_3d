import matplotlib
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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

# ================= 2. 字体与负号配置 =================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ================= 3. 座舱部件坐标定义 (已移除方向盘) =================
@dataclass
class DriverPosition:
    x: float = -450.0
    y: float = 800.0
    z: float = 700.0

# 🪟 挡风玻璃
WINDSHIELD_ORDERED = np.array([
    [-620,  900, 850], [ 620,  900, 850], [ 720, 1260, 220], [-720, 1260, 220]
])
WINDSHIELD_CLOSED = np.vstack([WINDSHIELD_ORDERED, WINDSHIELD_ORDERED[0]])

# 🪞 左后视镜
LEFT_MIRROR_ORDERED = np.array([
    [-980, 820, 480], [-900, 850, 480], [-900, 850, 380], [-980, 820, 380]
])
LEFT_MIRROR_CLOSED = np.vstack([LEFT_MIRROR_ORDERED, LEFT_MIRROR_ORDERED[0]])

# 🪞 右后视镜
RIGHT_MIRROR_ORDERED = np.array([
    [900, 850, 480], [980, 820, 480], [980, 820, 380], [900, 850, 380]
])
RIGHT_MIRROR_CLOSED = np.vstack([RIGHT_MIRROR_ORDERED, RIGHT_MIRROR_ORDERED[0]])

# 🪟 左侧窗 (门玻璃)
LEFT_WIN_ORDERED = np.array([
    [-880, 950, 720], [-880, 450, 680], [-880, 450, 200], [-880, 950, 200]
])
LEFT_WIN_CLOSED = np.vstack([LEFT_WIN_ORDERED, LEFT_WIN_ORDERED[0]])

# 🪟 右侧窗 (门玻璃)
RIGHT_WIN_ORDERED = np.array([
    [880, 950, 720], [880, 450, 680], [880, 450, 200], [880, 950, 200]
])
RIGHT_WIN_CLOSED = np.vstack([RIGHT_WIN_ORDERED, RIGHT_WIN_ORDERED[0]])

# ================= 4. 交互可视化核心类 =================
class InteractiveFOVViewer:
    def __init__(self, ws_ord, ws_cls, lm_ord, lm_cls, rm_ord, rm_cls, lw_ord, lw_cls, rw_ord, rw_cls):
        self.ws_ord, self.ws_cls = ws_ord, ws_cls
        self.lm_ord, self.lm_cls = lm_ord, lm_cls
        self.rm_ord, self.rm_cls = rm_ord, rm_cls
        self.lw_ord, self.lw_cls = lw_ord, lw_cls
        self.rw_ord, self.rw_cls = rw_ord, rw_cls
        self.driver = DriverPosition()

        self.fig = plt.figure(figsize=(14, 8), facecolor='#1a1a2e')
        self.ax = self.fig.add_subplot(121, projection='3d', facecolor='#16213e')
        self.ax2 = self.fig.add_subplot(122, facecolor='#16213e') # 笛卡尔坐标系
        plt.subplots_adjust(bottom=0.18, left=0.08, right=0.95, top=0.92, wspace=0.25)

        self.fov_lines = []
        self._setup_plot()
        self._setup_sliders()

    def _setup_plot(self):
        self.ax.set_title('座舱几何与视野包络 (无方向盘)', color='white', fontsize=14, pad=20)
        self.ax.set_xlabel('X 横向 (mm)', color='#aaa')
        self.ax.set_ylabel('Y 纵向 (mm)', color='#aaa')
        self.ax.set_zlabel('Z 垂向 (mm)', color='#aaa')

        self.ax.set_xlim(-1000, 1000)
        self.ax.set_ylim(200, 1400)
        self.ax.set_zlim(0, 1000)

        # 3D 部件渲染
        self.ax.plot(self.ws_cls[:,0], self.ws_cls[:,1], self.ws_cls[:,2], 'c-', linewidth=2, label='挡风玻璃')
        self.ax.add_collection3d(Poly3DCollection([list(zip(self.ws_ord[:,0], self.ws_ord[:,1], self.ws_ord[:,2]))],
                                                  facecolors='cyan', alpha=0.12, linewidths=0))
        self.ax.plot(self.lm_cls[:,0], self.lm_cls[:,1], self.lm_cls[:,2], '#ff9900', linewidth=2, label='左后视镜')
        self.ax.add_collection3d(Poly3DCollection([list(zip(self.lm_ord[:,0], self.lm_ord[:,1], self.lm_ord[:,2]))],
                                                  facecolors='#ff9900', alpha=0.5, linewidths=0))
        self.ax.plot(self.rm_cls[:,0], self.rm_cls[:,1], self.rm_cls[:,2], '#32cd32', linewidth=2, label='右后视镜')
        self.ax.add_collection3d(Poly3DCollection([list(zip(self.rm_ord[:,0], self.rm_ord[:,1], self.rm_ord[:,2]))],
                                                  facecolors='#32cd32', alpha=0.5, linewidths=0))
        self.ax.plot(self.lw_cls[:,0], self.lw_cls[:,1], self.lw_cls[:,2], '#8899aa', linewidth=1.5, label='左侧窗')
        self.ax.add_collection3d(Poly3DCollection([list(zip(self.lw_ord[:,0], self.lw_ord[:,1], self.lw_ord[:,2]))],
                                                  facecolors='#8899aa', alpha=0.15, linewidths=0))
        self.ax.plot(self.rw_cls[:,0], self.rw_cls[:,1], self.rw_cls[:,2], '#8899aa', linewidth=1.5, label='右侧窗')
        self.ax.add_collection3d(Poly3DCollection([list(zip(self.rw_ord[:,0], self.rw_ord[:,1], self.rw_ord[:,2]))],
                                                  facecolors='#8899aa', alpha=0.15, linewidths=0))

        for i, (color, label) in enumerate(zip(['r','g','b'], ['X','Y','Z'])):
            self.ax.plot([0, 300*(i==0)], [0, 300*(i==1)], [0, 300*(i==2)], color=color, linewidth=2, label=f'{label}轴')

        self.driver_point, = self.ax.plot([], [], [], 'wo', markersize=8, label='眉心')
        self.ax.legend(fontsize=8, facecolor='#0f3460', labelcolor='white', loc='upper left')
        self.ax.grid(True, alpha=0.3, color='#444')
        self.ax.set_box_aspect([1, 1.2, 0.6])

        # 2D 角度包络图配置
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
        ax_y = plt.axes([0.1, 0.08, 0.25, 0.04], facecolor='#0f3460')
        self.slider_y = Slider(ax_y, '前后 Y (mm)', 600, 1000, valinit=800, color=slider_color)
        self.slider_y.on_changed(self._update)

        ax_z = plt.axes([0.1, 0.03, 0.25, 0.04], facecolor='#0f3460')
        self.slider_z = Slider(ax_z, '高低 Z (mm)', 600, 800, valinit=700, color=slider_color)
        self.slider_z.on_changed(self._update)

        ax_reset = plt.axes([0.4, 0.04, 0.1, 0.05], facecolor='#e94560')
        self.btn_reset = Button(ax_reset, '重置', color='#c0392b', hovercolor='#e74c3c')
        self.btn_reset.on_clicked(self._reset)

        self.info_text = self.fig.text(0.52, 0.04, '', color='white', fontsize=8,
                                       bbox=dict(facecolor='#0f3460', edgecolor='#e94560', boxstyle='round,pad=0.4'))

    def _calc_angles(self, points):
        origin = np.array([self.driver.x, self.driver.y, self.driver.z])
        rel = points - origin
        r = np.linalg.norm(rel, axis=1)
        az = np.arctan2(rel[:, 0], rel[:, 1]) * 180/np.pi
        el = np.arctan2(rel[:, 2], np.sqrt(rel[:,0]**2 + rel[:,1]**2)) * 180/np.pi
        return r, az, el

    def _update(self, val=None):
        self.driver.y = self.slider_y.val
        self.driver.z = self.slider_z.val
        self.driver_point.set_data([self.driver.x], [self.driver.y])
        self.driver_point.set_3d_properties([self.driver.z])

        for line in self.fov_lines: line.remove()
        self.fov_lines.clear()
        origin = np.array([self.driver.x, self.driver.y, self.driver.z])

        # 绘制3D视线
        for pt, col, ls, alp in [(self.ws_ord, '#ffff00', '--', 0.4),
                                 (self.lm_ord, '#ff9900', '-', 0.6),
                                 (self.rm_ord, '#32cd32', '-', 0.6),
                                 (self.lw_ord, '#8899aa', ':', 0.3),
                                 (self.rw_ord, '#8899aa', ':', 0.3)]:
            for p in pt:
                self.fov_lines.append(self.ax.plot([origin[0], p[0]], [origin[1], p[1]], [origin[2], p[2]],
                                                   color=col, ls=ls, lw=0.8, alpha=alp)[0])

        # 计算所有部件角度
        _, az_ws, el_ws = self._calc_angles(self.ws_ord)
        _, az_lm, el_lm = self._calc_angles(self.lm_ord)
        _, az_rm, el_rm = self._calc_angles(self.rm_ord)
        _, az_lw, el_lw = self._calc_angles(self.lw_ord)
        _, az_rw, el_rw = self._calc_angles(self.rw_ord)

        # 更新2D角度包络图
        self.ax2.clear()
        self.ax2.set_title('全视野角度包络 (方位角-俯仰角)', color='white', fontsize=12, pad=15)
        self.ax2.set_xlabel('方位角 φ (°)', color='#aaa')
        self.ax2.set_ylabel('俯仰角 θ (°)', color='#aaa')
        self.ax2.set_xlim(-160, 160)
        self.ax2.set_ylim(-50, 45)
        self.ax2.grid(True, alpha=0.3, color='#444', linestyle=':')
        self.ax2.axhline(0, color='#aaa', linewidth=0.8)
        self.ax2.axvline(0, color='#aaa', linewidth=0.8)

        # ✅ 核心修复：质心极角排序法，确保四边形不自交
        def plot_env_safe(az, el, col, lab, alp=0.4):
            # 计算几何中心
            center = np.array([np.mean(az), np.mean(el)])
            # 计算各点相对于中心的角度
            angles = np.arctan2(el - center[1], az - center[0])
            # 按角度循环排序
            idx = np.argsort(angles)
            az_c = np.concatenate([az[idx], [az[idx[0]]]])
            el_c = np.concatenate([el[idx], [el[idx[0]]]])
            self.ax2.fill(az_c, el_c, color=col, alpha=alp, label=lab)
            self.ax2.plot(az_c, el_c, color=col, lw=1.5)

        plot_env_safe(az_ws, el_ws, '#e94560', '前向视野', 0.3)
        plot_env_safe(az_lm, el_lm, '#ff9900', '左后视镜', 0.5)
        plot_env_safe(az_rm, el_rm, '#32cd32', '右后视镜', 0.5)
        plot_env_safe(az_lw, el_lw, '#8899aa', '左侧窗', 0.4)
        plot_env_safe(az_rw, el_rw, '#8899aa', '右侧窗', 0.4)
        self.ax2.legend(fontsize=9, facecolor='#0f3460', labelcolor='white', loc='upper right')

        self.info_text.set_text(
            f"眉心: ({self.driver.x:+.0f}, {self.driver.y:+.0f}, {self.driver.z:+.0f}) mm\n"
            f"前挡: φ[{az_ws.min():+.1f}°~{az_ws.max():+.1f}°]  左镜: φ[{az_lm.min():+.1f}°~{az_lm.max():+.1f}°]\n"
            f"右镜: φ[{az_rm.min():+.1f}°~{az_rm.max():+.1f}°]  左窗: φ[{az_lw.min():+.1f}°~{az_lw.max():+.1f}°]\n"
            f"右窗: φ[{az_rw.min():+.1f}°~{az_rw.max():+.1f}°]"
        )
        self.fig.canvas.draw_idle()

    def _reset(self, event):
        self.slider_y.reset()
        self.slider_z.reset()

    def show(self):
        self._update()
        if BACKEND_NAME == 'Agg':
            save_path = 'cockpit_clean_envelope.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=self.fig.get_facecolor())
            print(f"🖼️  已保存至: {os.path.abspath(save_path)}")
        else:
            plt.show()

# ================= 5. 主程序入口 =================
if __name__ == "__main__":
    viewer = InteractiveFOVViewer(
        WINDSHIELD_ORDERED, WINDSHIELD_CLOSED,
        LEFT_MIRROR_ORDERED, LEFT_MIRROR_CLOSED,
        RIGHT_MIRROR_ORDERED, RIGHT_MIRROR_CLOSED,
        LEFT_WIN_ORDERED, LEFT_WIN_CLOSED,
        RIGHT_WIN_ORDERED, RIGHT_WIN_CLOSED
    )
    viewer.show()