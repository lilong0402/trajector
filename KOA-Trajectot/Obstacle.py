import numpy as np
from flatbuffers.flexbuffers import Object


class Obstacle(Object):
    def __init__(self):
        self.sphere_coordinate = (0,0,0)
        self.sphere_radius = 1
    # 构造椭球形障碍物
    """
        traj_point_corrdinate : 轨迹点坐标
        a,b,c 椭球体沿着三个坐标轴的半轴长度
    """
    def obs_ellipsoid(traj_point_corrdinate, a, b, c):
        return (traj_point_corrdinate[0] ** 2 / a ** 2) + (traj_point_corrdinate[1] ** 2 / b ** 2) + (
                    traj_point_corrdinate[2] ** 2 / c ** 2) < 1

    # 构造圆锥障碍物
    """
        traj_point_corrdinate : 轨迹点坐标
        conical_tip_coordinate: 圆锥顶点坐标
        h,r : 圆锥体高度、地面半径
    """
    def obs_concial(traj_point_corrdinate, conical_tip_coordinate, h, r):
        return ((traj_point_corrdinate[0] - conical_tip_coordinate[0]) ** 2 + (
                    traj_point_corrdinate[1] - conical_tip_coordinate[1]) ** 2) / r ** 2 < (
                    traj_point_corrdinate[2] - conical_tip_coordinate[2]) / h

    # 构造球形障碍物
    """
        trajector : 轨迹
        sphere_centre_coordinate : 球心坐标
        radius : 半径
    """
    def obs_sphere(self, trajector, sphere_centre_coordinate, radius):
        self.sphere_coordinate = sphere_centre_coordinate
        self.sphere_radius = radius
        for i in range(len(trajector) - 1):
            point1, point2 = trajector[i:i + 2]
            A = (point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2 + (point2[2] - point1[2]) ** 2
            B = 2 * ((point2[0] - point1[0]) * (point1[0] - sphere_centre_coordinate[0]) + (point2[1] - point1[1]) * (
                        point1[1] - sphere_centre_coordinate[1]) + (point2[2] - point1[2]) * (
                                 point1[2] - sphere_centre_coordinate[2]))
            C = (point1[0] - sphere_centre_coordinate[0]) ** 2 + (point1[1] - sphere_centre_coordinate[1]) ** 2 + (
                        point1[2] - sphere_centre_coordinate[2]) ** 2 - radius ** 2
            D = B ** 2 - 4 * A * C
            if D < 0:
                continue
                # 处理A接近0的情况（线段长度极短）
            if np.isclose(A, 0):
                # 直接检查线段端点是否在球内
                if (point1[0] - sphere_centre_coordinate[0]) ** 2 + (point1[1] - sphere_centre_coordinate[1]) ** 2 + (
                        point1[2] - sphere_centre_coordinate[2]) ** 2 <= radius ** 2:
                    return True
                continue

            # 检查交点是否在线段范围内 (t ∈ [0, 1])
            sqrt_D = np.sqrt(D)
            t1 = (-B + sqrt_D) / (2 * A)
            t2 = (-B - sqrt_D) / (2 * A)
            if (0 <= t1 <= 1) or (0 <= t2 <= 1):
                return True  # 线段与球体相交
        for point in trajector:
            return (point[0]-self.sphere_coordinate[0]) ** 2 + (point[1]-self.sphere_coordinate[1]) ** 2 + (point[2]-self.sphere_coordinate[2]) ** 2 <=  radius ** 2
        return False  # 所有线段均未与球体相交

    # 与球体障碍物的间距与安全水平
    def dist_sphere(self, trajector_point_coordinate,safe_dist=1):
        trajector_point_coordinate = np.array(trajector_point_coordinate)
        sphere_coordinate = np.array(self.sphere_coordinate)
        dist = np.linalg.norm(trajector_point_coordinate - sphere_coordinate) - self.sphere_radius


        if dist < safe_dist:
            return dist / safe_dist
        return 1


