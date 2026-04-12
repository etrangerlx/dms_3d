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
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ================= 3. 数据结构定义 =================
@dataclass
class DriverPosition:
    x: float = -450.0
    y: float = 800.0
    z: float = 700.0

# --- A. 挡风玻璃 (青色) ---
WINDSHIELD_ORDERED = np.array([
    [-650, 1150, 820], [-720, 1100, 450], [-700, 1300, 180],
    [  0 , 1350, 200], [ 700, 1300, 180], [ 720, 1100, 450],
    [ 650, 1150, 820], [  0 , 1200, 850],
])
WINDSHIELD_CLOSED = np.vstack([WINDSHIELD_ORDERED, WINDSHIELD_ORDERED[0]])

# --- B. 左后视镜 (橙色) ---
LEFT_MIRROR_ORDERED = np.array([
    [-980, 820, 480], [-900, 850, 480], [-900, 850, 380], [-980, 820, 380],
])
LEFT_MIRROR_CLOSED = np.vstack([LEFT_MIRROR_ORDERED, LEFT_MIRROR_ORDERED[0]])

# --- C. 右后视镜 (亮绿色) ---
RIGHT_MIRROR_ORDERED = np.array([
    [900, 850, 480], [980, 820, 480], [980, 820, 380], [900, 850, 380],
])
RIGHT_MIRROR_CLOSED = np.vstack([RIGHT_MIRROR_ORDERED, RIGHT_MIRROR_ORDERED[0]])

# ================= 4. 交互可视化核心类 =================
class InteractiveFOVViewer:
    def __init__(self, ws_ordered, ws_closed, lm_ordered, lm_closed, rm_ordered, rm_closed):
        self.ws_ordered = ws_ordered
        self.ws_closed = ws_closed
        self.lm_ordered = lm_ordered
        self.lm_closed = lm_closed
        self.rm_ordered = rm_ordered
        self.rm_closed = rm_closed
        self.driver = DriverPosition()

        self.fig = plt.figure(figsize=(14, 8), facecolor='#1a1a2e')
        self.ax = self.fig.add_subplot(121, projection='3d', facecolor='#16213e')
        self.ax2 = self.fig.add_subplot(122, projection='polar', facecolor='#16213e')
        plt.subplots_adjust(bottom=0.18, left=0.08, right=0.95, top=0.92, wspace=0.25)

        self.fov_lines = []
        self.ws_poly = self.lm_poly = self.rm_poly = None

        self._setup_plot()
        self._setup_sliders()

    def _setup_plot(self):
        self.ax.set_title('驾驶员全视野包络 (前+左+右)', color='white', fontsize=14, pad=20)
        self.ax.set_xlabel('X 横向 (mm)', color='#aaa')
        self.ax.set_ylabel('Y 纵向 (mm)', color='#aaa')
        self.ax.set_zlabel('Z 垂向 (mm)', color='#aaa')

        self.ax.set_xlim(-1100, 1100)
        self.ax.set_ylim(600, 1400)
        self.ax.set_zlim(0, 1000)

        # 1. 绘制挡风玻璃
        self.ax.plot(self.ws_closed[:,0], self.ws_closed[:,1], self.ws_closed[:,2], 'c-', linewidth=2, label='挡风玻璃')
        self.ws_poly = Poly3DCollection([list(zip(self.ws_ordered[:,0], self.ws_ordered[:,1], self.ws_ordered[:,2]))],
                                        facecolors='cyan', linewidths=1, edgecolors='c', alpha=0.1)
        self.ax.add_collection3d(self.ws_poly)

        # 2. 绘制左后视镜
        self.ax.plot(self.lm_closed[:,0], self.lm_closed[:,1], self.lm_closed[:,2], color='#ff9900', linewidth=2, label='左后视镜')
        self.lm_poly = Poly3DCollection([list(zip(self.lm_ordered[:,0], self.lm_ordered[:,1], self.lm_ordered[:,2]))],
                                        facecolors='#ff9900', linewidths=1, edgecolors='#ff9900', alpha=0.6)
        self.ax.add_collection3d(self.lm_poly)

        # 3. 绘制右后视镜
        self.ax.plot(self.rm_closed[:,0], self.rm_closed[:,1], self.rm_closed[:,2], color='#32cd32', linewidth=2, label='右后视镜')
        self.rm_poly = Poly3DCollection([list(zip(self.rm_ordered[:,0], self.rm_ordered[:,1], self.rm_ordered[:,2]))],
                                        facecolors='#32cd32', linewidths=1, edgecolors='#32cd32', alpha=0.6)
        self.ax.add_collection3d(self.rm_poly)

        # 坐标轴
        for i, (color, label) in enumerate(zip(['r','g','b'], ['X','Y','Z'])):
            self.ax.plot([0, 400*(i==0)], [0, 400*(i==1)], [0, 400*(i==2)], color=color, linewidth=2, label=f'{label}轴')

        self.driver_point, = self.ax.plot([], [], [], 'wo', markersize=8, label='眉心')
        self.ax.legend(fontsize=8, facecolor='#0f3460', labelcolor='white')
        self.ax.grid(True, alpha=0.3, color='#444')
        self.ax.set_box_aspect([1, 1.2, 0.6])

        # 极坐标配置
        self.ax2.set_title('全视野角度包络 (方位角-俯仰角)', color='white', fontsize=12, pad=15)
        self.ax2.set_theta_zero_location('N')  # 0度在正上方 (Y轴)
        self.ax2.set_theta_direction(-1)       # 顺时针为正 (符合车辆坐标系：右正左负)
        self.ax2.tick_params(colors='#aaa')
        self.ax2.set_rlim(-45, 45)             # 径向：俯仰角 -45° 到 +45°

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

        self.info_text = self.fig.text(0.52, 0.04, '', color='white', fontsize=8.5,
                                       bbox=dict(facecolor='#0f3460', edgecolor='#e94560', boxstyle='round,pad=0.4'))

    def _calc_angles(self, points):
        """通用函数：计算任意点集相对于眉心的方位角和俯仰角"""
        origin = np.array([self.driver.x, self.driver.y, self.driver.z])
        relative = points - origin
        # 距离
        r = np.linalg.norm(relative, axis=1)
        # 方位角 (Azimuth): atan2(x, y), 范围 -180 ~ 180
        azimuth = np.arctan2(relative[:, 0], relative[:, 1]) * 180/np.pi
        # 俯仰角 (Elevation): atan2(z, sqrt(x^2+y^2)), 范围 -90 ~ 90
        elevation = np.arctan2(relative[:, 2], np.sqrt(relative[:,0]**2 + relative[:,1]**2)) * 180/np.pi
        return r, azimuth, elevation

    def _update(self, val=None):
        self.driver.y = self.slider_y.val
        self.driver.z = self.slider_z.val

        # 更新眉心
        self.driver_point.set_data([self.driver.x], [self.driver.y])
        self.driver_point.set_3d_properties([self.driver.z])

        # 清除旧视线
        for line in self.fov_lines: line.remove()
        self.fov_lines.clear()

        origin = np.array([self.driver.x, self.driver.y, self.driver.z])

        # --- 1. 计算并绘制 前挡风玻璃 ---
        r_ws, az_ws, el_ws = self._calc_angles(self.ws_ordered)
        for pt in self.ws_ordered:
            self.fov_lines.append(self.ax.plot([origin[0], pt[0]], [origin[1], pt[1]], [origin[2], pt[2]], 'y--', linewidth=0.6, alpha=0.4)[0])

        # --- 2. 计算并绘制 左后视镜 ---
        r_lm, az_lm, el_lm = self._calc_angles(self.lm_ordered)
        for pt in self.lm_ordered:
            self.fov_lines.append(self.ax.plot([origin[0], pt[0]], [origin[1], pt[1]], [origin[2], pt[2]], '#ff9900', linewidth=0.8, alpha=0.6)[0])

        # --- 3. 计算并绘制 右后视镜 ---
        r_rm, az_rm, el_rm = self._calc_angles(self.rm_ordered)
        for pt in self.rm_ordered:
            self.fov_lines.append(self.ax.plot([origin[0], pt[0]], [origin[1], pt[1]], [origin[2], pt[2]], '#32cd32', linewidth=0.8, alpha=0.6)[0])

        # ================= 更新极坐标图 (核心修改) =================
        self.ax2.clear()
        self.ax2.set_theta_zero_location('N')
        self.ax2.set_theta_direction(-1)
        self.ax2.set_title('全视野角度包络', color='white', fontsize=12, pad=15)
        self.ax2.tick_params(colors='#aaa')
        self.ax2.set_rlim(-45, 45)
        self.ax2.set_yticks(np.arange(-40, 45, 10))
        self.ax2.grid(True, alpha=0.3, color='#444', linestyle=':')

        # 辅助函数：绘制极坐标包络
        def plot_polar_envelope(az, el, color, label, alpha=0.4):
            # 闭合多边形
            az_closed = np.concatenate([az, az[:1]])
            el_closed = np.concatenate([el, el[:1]])
            # 角度转弧度 (注意 matplotlib polar 使用弧度)
            theta = np.deg2rad(az_closed)
            # 绘制填充
            self.ax2.fill(theta, el_closed, color=color, alpha=alpha, label=label)
            self.ax2.plot(theta, el_closed, color=color, linewidth=1.5)

        # 绘制三个区域
        plot_polar_envelope(az_ws, el_ws, '#e94560', '前向视野', alpha=0.3) # 红
        plot_polar_envelope(az_lm, el_lm, '#ff9900', '左后视镜', alpha=0.5) # 橙
        plot_polar_envelope(az_rm, el_rm, '#32cd32', '右后视镜', alpha=0.5) # 绿

        self.ax2.legend(fontsize=9, facecolor='#0f3460', labelcolor='white', loc='lower left')

        # 更新信息面板
        info = (f"眉心: ({self.driver.x:+.0f}, {self.driver.y:+.0f}, {self.driver.z:+.0f}) mm\n"
                f"前向: φ[{az_ws.min():+.1f}°, {az_ws.max():+.1f}°] θ[{el_ws.min():+.1f}°, {el_ws.max():+.1f}°]\n"
                f"左镜: φ[{az_lm.min():+.1f}°, {az_lm.max():+.1f}°] θ[{el_lm.min():+.1f}°, {el_lm.max():+.1f}°]\n"
                f"右镜: φ[{az_rm.min():+.1f}°, {az_rm.max():+.1f}°] θ[{el_rm.min():+.1f}°, {el_rm.max():+.1f}°]")
        self.info_text.set_text(info)

        self.fig.canvas.draw_idle()

    def _reset(self, event):
        self.slider_y.reset()
        self.slider_z.reset()

    def show(self):
        self._update()
        if BACKEND_NAME == 'Agg':
            save_path = 'fov_full_envelope.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=self.fig.get_facecolor())
            print(f"🖼️  已保存至: {os.path.abspath(save_path)}")
        else:
            plt.show()

# ================= 5. 主程序入口 =================
if __name__ == "__main__":
    viewer = InteractiveFOVViewer(
        WINDSHIELD_ORDERED, WINDSHIELD_CLOSED,
        LEFT_MIRROR_ORDERED, LEFT_MIRROR_CLOSED,
        RIGHT_MIRROR_ORDERED, RIGHT_MIRROR_CLOSED
    )
    viewer.show()