# 环境建模

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：machineLearning
@File    ：KOA.py
@IDE     ：PyCharm
@Author  ：苜李龙
@Date    ：2025/4/20 22:33
'''
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
import Obstacle
from Utils import dist
# 随机种子
np.random.seed(0)

ob = Obstacle.Obstacle()

# 环境建模


#    约束条件
# 飞行高度约束 [L-low,L-high]

# 最大航程约束  <L-max

# 转弯角约束  [0,B-max]

# 地形约束 避障

# 最大俯仰角约束

# 无人机速度约束
"""
    无人机致死率-无人机坠落时的重量核高度与事故致死率有关
        FR = P1 * P2 *P3
        P1 P2 P3 分别代表 无人机坠落或失控的概率  无人机坠落地点存在地面人员的概率 无人机命中地面人员后造成死亡的概率
"""

# 空域拥挤度




# 构建适应度函数
"""
    trajector: 候选解
    obs : 障碍物
        obs : [[obs_name,[obs_paramater[圆心、半径]]]]
    w  :各种惩罚系数
    safe_dist : 安全距离
    fatality_rate : 致死率
"""
def fitness_func(trajectors,obs,w,safe_dist,fatality_rate):
    trajectors_length = []
    trajectors_safe = []
    # 各个航迹的长度
    for trajector in trajectors:
        trajector_length = 0
        trajector_safe = np.zeros(len(obs))
        for i,point in zip(range(len(trajector)-1),trajector):
            trajector_length += dist(trajector[i], trajector[i+1])
            # 计算航迹点与所有障碍物的安全水平
            for j in range(len(obs)):
                if obs[j][0] == "sphere":
                    trajector_safe[j] += ob.dist_sphere(point,safe_dist)
                    break
        trajectors_length.append(trajector_length)
        # 航迹点对不同障碍物的安全水平 (0,1)
        trajectors_safe.append(trajector_safe)

    # 对航段成本归一化处理防止其值过大而对结果产生较大的影响
    max_length = max(trajectors_length)
    min_length = min(trajectors_length)
    trajectors_length_temp = [(max_length - i)/(max_length - min_length) for i in trajectors_length]
    trajectors_length = trajectors_length_temp
    del trajectors_length_temp

    tj_size = len(trajectors_safe)
    trajectors_safe = np.array(trajectors_safe).reshape(-1)
    trajectors_safe /= tj_size
    trajectors_length = np.array(trajectors_length).reshape(-1)

    # print(trajectors_length)
    # print(trajectors_safe)

    fit = w[0] * trajectors_length + w[1] * trajectors_safe + w[2] * fatality_rate
    return fit
"""
    start: 起始点
    end： 终点
    num_trajector: 航迹数量
    num_trajector_point: 航迹点数量
"""
def inittrajector(start,end,num_trajector,num_trajector_point):
    trajectors = []
    for i in range(num_trajector):
        a = [start[0],start[1],start[2]]
        b = end
        trajector=[start]
        inteverval = [(end[0]-start[0])/ num_trajector_point, (end[1]-start[1])/ num_trajector_point, (end[2]-start[2])/ num_trajector_point]
        for j in range(num_trajector_point):
            (x,y,z) = (np.random.uniform(a[0],a[0] + inteverval[0]),np.random.uniform(a[1],a[1] + inteverval[1]),np.random.uniform(a[2],a[2] + inteverval[2]))
            a = a + inteverval
            trajector.append((x,y,z))
        trajector.append(b)
        trajectors.append(trajector)
    return trajectors

#    计算引力,并更新位置
"""
    exfitness : 最优适应度的值
    fit_index : 最优候选解索引
    trajectors : 所有候选解
    a : 随机扰动强度 0.01-0.5 随迭代次数递减
    G : 引力常数
    q : 防止除0错误 10 ** -6
    Levy Flight : 随机扰动项(避免局部最优)
    t = 1  虚拟时间步长
"""
def gravitation(exfitness,fit_index,trajectors,a,G=1,q=10 ** -6,LevyFlight=0.1,t = 1):
    ex_trajector = trajectors[fit_index]
    ex_trajector = np.array(ex_trajector)
    for trajector in trajectors:
        for i,point in enumerate(trajector):
            if i == 0 :
                continue
            F =  (G * (exfitness/3)/((point[0] - ex_trajector[i][0]) ** 2 + (point[1] - ex_trajector[i][1]) **2 + (point[2] - ex_trajector[i][2]) ** 2 + q)
                  * np.array([ex_trajector[i][0] - point[0],ex_trajector[i][1] - point[1],ex_trajector[i][2] - point[2]]))
            trajector[i] = point + t * F + a * LevyFlight
# KOA 优化算法
"""
    start: 起始点
    end： 终点
    obs_parameter : 障碍物参数
    num_trajector: 航迹数量
    num_trajector_point: 航迹点数量
    safe_dist : 安全目标水平 默认 1米
    a : 随机扰动强度 0.01-0.5 随迭代次数递减
    G : 引力常数
    q : 防止除0错误 10 ** -6
    Levy Flight : 随机扰动项(避免局部最优)
    t = 1  虚拟时间步长
    epochs: 迭代次数默认 100 
"""
def KOA(start,end,obs,w,fatality_rate,num_trajector=10,num_trajector_point=5,safe_dist = 1,G = 1,q = 10 ** -6,LevyFlight=1,t = 1,epochs = 100):
    num_trajector_point -=2
#  初始行星种群（航迹）
    trajectors = inittrajector(start,end,num_trajector,num_trajector_point)
    # print(trajectors)
    for j in range(len(obs)):
        if obs[j][0] == "sphere":
            for trajector in trajectors:
                if ob.obs_sphere(trajector, obs[j][1][0], obs[j][1][1]):
                    trajectors.remove(trajector)
    # print(f"trajectors : {len(trajectors)} ")
    # 计算适应度
    fits = fitness_func(trajectors,obs,w,safe_dist,fatality_rate)

    # 我们获取适应度最好的航段的索引
    fit_index = np.argmax(fits)
    # print(f"fit_index:{fit_index}")
    #
    # print(f"fits = {fits}")
    # print(len(fits))
#  计算引力和速度
#   根据万有引力公式，计算各行星所受引力的大小，其中的太阳质量和行星质量由适应度函数决定
#  速度取决于行星相对于太阳的位置，与太阳越近，引力增大，速度增快，反之速度下降（利用距离太阳较远的行星进行探索，以寻找新的解决方案，利用距离太阳教员的行星进行开发以搜索最优解附近的新位置）
# 位置更新与保留
    a = 0.5
    for epoch in range(epochs):
        a -= (0.45/epochs)
        gravitation(fits[fit_index],fit_index,trajectors,a,G,q,LevyFlight,t)
        fits = fitness_func(trajectors, obs, w, safe_dist, fatality_rate)
        fit_index = np.argmax(fits)
# 画出最优路径的图形
    return trajectors,fit_index

if __name__ == "__main__":
    start = (0,0,0)
    end = (100,100,0)
    num_trajector = 1000
    num_trajector_point = 15
    obs = [["sphere",[(30,30,30),0]]]
    fatality_rate = 0
    w = [2,1,1]
    epochs = 100

    trajectors,fit_index = KOA(start=start,end = end,obs = obs,w = w,fatality_rate = fatality_rate,num_trajector = num_trajector,num_trajector_point = num_trajector_point,epochs=epochs)
    center = obs[0][1][0]
    R = obs[0][1][1]

    # 参数化网格
    u = np.linspace(0, 2 * np.pi, 100)  # 经度角
    v = np.linspace(0, np.pi, 100)  # 纬度角
    # 生成球面坐标
    x = center[0] + R * np.outer(np.cos(u), np.sin(v))
    y = center[1] + R * np.outer(np.sin(u), np.sin(v))
    z = center[2] + R * np.outer(np.ones(np.size(u)), np.cos(v))

    # 绘制球体
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, color='b', alpha=0.1, edgecolor='k')
    #
    # for trajector in trajectors:
    #     a_x = [point[0] for point in trajector]
    #     a_y = [point[1] for point in trajector]
    #     a_z = [point[2] for point in trajector]
    #     ax.plot(a_x, a_y, a_z, color='r')
    trajector = trajectors[fit_index]
    a_x = [point[0] for point in trajector]
    a_y = [point[1] for point in trajector]
    a_z = [point[2] for point in trajector]
    ax.plot(a_x,a_y,a_z,color='b')
    plt.show()
    print(f"最优航迹路线:{trajector}")
    # for i in range(len(trajectors)):
    #     print(trajectors[i])
    #     print("\n")
