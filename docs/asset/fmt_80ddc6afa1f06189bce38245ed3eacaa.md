
```py
from optalgotools.algorithms import SimulatedAnnealing
from optalgotools.problems import TSP
import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx
import folium
import folium.plugins

wal_df = pd.read_csv("https://raw.githubusercontent.com/Optimization-Algorithms-
➥Book/Code-Listings/main/Appendix%20B/data/TSP/Walmart_ON.csv")                     ①

cities_list = [city for city, region in cityToRegion.items() if city in
➥wal_df.city.unique() and region in ['Durham Region', 'York Region', 'Toronto']]    ②

gta_part = wal_df[wal_df.store_number.str.startswith('Walmart Supercentre') &
➥ wal_df.city.isin(cities_list)].reset_index(drop=True)                             ③
wal_gta_count = gta_part.shape[0]

gta_part_loc = gta_part[['latitude', 'longitude']]                                   ④

G = ox.graph_from_point(tuple(gta_part_loc.mean().to_list()), dist=42000,            ④
➥ dist_type='network', network_type='drive', clean_periphery=True, simplify=True,   ④
➥ retain_all=True, truncate_by_edge=True)                                           ④

gta_part['osmid'], gta_part['osmid_dist_m'] = zip(*gta_part.apply(lambda row:
➥ox.nearest_nodes(G, row.longitude, row.latitude, return_dist=True), axis = 1)) 

gta_part_dists = np.zeros([wal_gta_count, wal_gta_count])                            ⑤
gta_part_pathes = [[[] for i in range(wal_gta_count)] for j in range(wal_gta_count)] ⑤
for i in range(wal_gta_count):                                                       ⑤
    for j in range(wal_gta_count):                                                   ⑤
        if i==j:                                                                     ⑤
            continue                                                                 ⑤
        gta_part_pathes[i][j] = nx.shortest_path(G=G, source=gta_part.osmid[i],      ⑤
➥ target=gta_part.osmid[j], weight='length', method='dijkstra')                     ⑤
        gta_part_dists[i][j] = nx.shortest_path_length(G=G,                          ⑤
➥source=gta_part.osmid[i], target=gta_part.osmid[j], weight='length',               ⑤
➥ method='dijkstra')/1000                                                           ⑤

gta_part_tsp = TSP(dists=gta_part_dists, gen_method='mutate')                        ⑥

sa = SimulatedAnnealing(max_iter=1000, max_iter_per_temp=100, initial_temp=85,
➥ final_temp=0.0001, cooling_schedule='linear')                                     ⑦

sa.init_annealing(gta_part_tsp)                                                      ⑧
sa.run(gta_part_tsp)                                                                 ⑨
```

① 加载安大略省所有沃尔玛门店的列表。

② 选择位于达勒姆地区、约克地区或多伦多的城市。

③ 选择前述列表中的沃尔玛门店，并且是超级中心。

④ 获取前述沃尔玛门店的经纬度位置，并创建一个连接这些门店且在 42 公里范围内的道路图。

⑤ 使用图计算沃尔玛门店之间的距离。

⑥ 为问题创建一个 `TSP` 对象。

⑦ 创建一个 `SA` 对象以帮助解决 `TSP` 问题。

⑧ 获取一个初始随机解，并检查其长度。

⑨ 运行 `SA`，并评估最佳解的距离。

正如你所见，我们在 `optalgotools` 中将求解器类与问题对象分离。求解器来自 `algorithms`，而问题是 `problems` 类中 `TSP` 问题的一个实例。这种实现允许你更改问题实例并调整算法的参数，以达到最优或近似最优解。你可以考虑尝试第 5.2.5 节中解释的 `SA` 的适应性方面，以了解它们对算法性能的影响，包括获得路线的长度和运行时间（`CPU` 时间和墙钟时间）。

类似于 `SA` 的元启发式算法在合理的计算成本下寻求最优或近似最优解，但不能保证它们的可行性或优化程度。通过适当的参数调整，算法可以提供可接受的解，而无需进一步的后处理。在下一章中，我们将讨论禁忌搜索作为另一种基于轨迹的优化算法。

## 摘要

+   元启发式算法可以大致分为基于轨迹的算法和基于种群的算法。基于轨迹的元启发式算法，或称为 `S` 元启发式算法，使用单个搜索代理以分段方式在设计或搜索空间中移动。基于种群的算法，或称为 `P` 元启发式算法，使用多个代理来寻找最优或近似最优的全局解。模拟退火是一种基于轨迹的元启发式算法。

+   模拟退火模仿材料处理中的退火过程，其中金属冷却并冻结成具有最小能量和较大晶体尺寸的晶体状态，以减少金属结构中的缺陷。退火过程涉及对温度和冷却速率的仔细控制，通常称为退火计划。

+   模拟退火在不同的热力学条件下运行一系列移动，并且总是接受改进的移动，并且可以以概率接受非改进的移动。

+   接受概率与温度成正比。高温增加了在搜索初期接受非改进移动的机会，以有利于搜索空间的探索。随着搜索的进行，温度递减以限制探索并有利于利用。

+   随着温度趋近于零，`SA` 表现得像爬山法一样贪婪，而当温度趋近于无穷大时，`SA` 表现得像随机游走。温度应逐渐降低，以实现探索和利用之间的最佳权衡。

+   模拟退火是一种随机搜索算法和无导数求解器，可以在导数信息不可用、不可靠或过于昂贵的情况下使用。模拟退火在合理的计算成本下寻求最优或近似最优解，但它不能保证解的可行性或最优程度。

+   自适应模拟退火可以根据搜索进程动态改变其参数，以控制探索和开发行为。

+   模拟退火是一种易于实现的概率近似算法，可用于解决不同领域中的连续和离散问题。
