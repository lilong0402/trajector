import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def initialize_trajectories(start, end, num_trajectories=100, num_waypoints=20,
                            height_range=(50, 150), max_deviation=0.5):
    """
    无人机随机航迹初始化函数（适合后续优化）

    参数：
        start: 起飞点 [x, y, z] (z建议≥0)
        end: 目标点 [x, y, z]
        num_trajectories: 航迹数量 (默认100)
        num_waypoints: 每条航迹的航路点数 (默认20)
        height_range: 高度范围 (米)
        max_deviation: 最大水平偏移比例 (0-1)

    返回：
        trajectories: 三维航迹数组 [num_trajectories, num_waypoints+2, 3]
    """
    # 参数校验
    assert len(start) == 3 and len(end) == 3, "坐标必须是三维的"
    assert max_deviation > 0 and max_deviation <= 1, "偏移比例应在0-1之间"

    start = np.array(start)
    end = np.array(end)
    trajectories = np.zeros((num_trajectories, num_waypoints + 2, 3))

    # 计算主方向向量和水平距离
    direction = end - start
    horizontal_dist = np.linalg.norm(direction[:2])
    vertical_dist = abs(end[2] - start[2])

    for i in range(num_trajectories):
        # 1. 固定起点和终点
        trajectories[i, 0] = start
        trajectories[i, -1] = end

        # 2. 生成中间航路点
        for j in range(1, num_waypoints + 1):
            # 线性插值基准
            t = j / (num_waypoints + 1)
            baseline = start + t * direction

            # 水平随机偏移（椭圆约束）
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, max_deviation * horizontal_dist)
            offset_x = radius * np.cos(angle)
            offset_y = radius * np.sin(angle)

            # 高度变化（考虑起点和目标点高度差）
            if end[2] >= start[2]:
                height = np.random.uniform(
                    start[2] + t * vertical_dist * 0.8,
                    min(start[2] + t * vertical_dist * 1.2, height_range[1])
                )
            else:
                height = np.random.uniform(
                    max(end[2] + (1 - t) * vertical_dist * 1.2, height_range[0]),
                    start[2] - t * vertical_dist * 0.8
                )

            # 应用偏移
            trajectories[i, j, 0] = baseline[0] + offset_x
            trajectories[i, j, 1] = baseline[1] + offset_y
            trajectories[i, j, 2] = np.clip(height, height_range[0], height_range[1])

    return trajectories


# 使用示例
if __name__ == "__main__":
    # 定义起降点（示例：从地面起飞，目标点高度100米）
    start_point = np.array([0, 0, 0])  # 地面起飞
    end_point = np.array([1000, 800, 100])  # 目标点

    # 生成100条航迹，每条22个点（20个航路点+起终点）
    trajectories = initialize_trajectories(
        start=start_point,
        end=end_point,
        num_trajectories=100,
        num_waypoints=20,
        height_range=(50, 150),  # 飞行高度50-150米
        max_deviation=0.4  # 水平最大偏移40%
    )

    # 3D可视化
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制所有航迹
    for traj in trajectories:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', alpha=0.15, linewidth=0.8)

    # 标注起终点
    ax.scatter(start_point[0], start_point[1], start_point[2],
               c='green', s=100, label='Start (Takeoff)', marker='^')
    ax.scatter(end_point[0], end_point[1], end_point[2],
               c='red', s=100, label='End (Target)', marker='s')

    # 绘制一条示例航迹（用粗线突出显示）
    example_idx = np.random.randint(100)
    ax.plot(trajectories[example_idx, :, 0],
            trajectories[example_idx, :, 1],
            trajectories[example_idx, :, 2],
            'r-', linewidth=2, label='Example Trajectory')

    # 设置图形属性
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title(f'Initialized {len(trajectories)} UAV Trajectories\n'
                 f'({trajectories.shape[1] - 2} waypoints per trajectory)')
    ax.legend()

    # 添加俯视和侧视角
    # ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.show()