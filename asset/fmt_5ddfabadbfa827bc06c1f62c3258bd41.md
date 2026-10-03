
```py
for ant in tqdm(range(n)):
    frontier = [origin_node]                                            ①
    explored = set()
    route = []
    found = False

    while frontier and not found:
        parent = frontier.pop(0)
        explored.add(parent)

        children = []
        children_pheremones = []
        for child in parent.expand():
            if child == destination_node:
                found = True
                route = child.path()
                continue
            if child not in explored:
                children.append(child)
                children_pheremones.append(
                    pheremone(
                        pheremone_concentrations[(parent.osmid,
➥ child.osmid)],
                        child.distance,
                        alpha,
                        beta,
                    )
                )

        if len(children) == 0:
            continue  

        transition_probability = [
            children_pheremones[i] / sum(children_pheremones)
            for i in range(len(children_pheremones))
        ]

        chosen = random.choices(children, weights=transition_probability,
➥ k=1)[0]                                                             ②

        children.pop(children.index(chosen))                           ③
        frontier.extend(children)                                      ③

        frontier.insert(0, chosen)                                     ④

    for u, v in zip(route[:-1], route[1:]):                            ⑤
        length_of_edge = G[u][v][0]['length']
        pheremone_concentrations[(u,v)] += Q/length_of_edge

    route = tuple(route)
    if route in known_routes:                                          ⑥
        known_routes[route] += 1
    else:
        known_routes[route] = 1
```

① 将蚂蚁放置在原点节点。

② 以概率选择一个子节点进行探索。

③ 添加所有未探索的子节点，以防我们以后需要探索它们。

④ 将选定的子节点设为下一个要探索的节点。

⑤ 更新信息素。

⑥ 如果路线是新发现的，则将其添加到列表中。

您现在可以打印最佳路线及其成本，并如下可视化获得的路线：

```py
best_route = max(known_routes, key=known_routes.get)
times_used = known_routes[best_route]
route = list(best_route)
print("Cost:", cost(G,route))
print("Times used:",times_used)
draw_route(G,route)
```

图 C.35 显示了 ACO 生成的最优路线。

![](img/APPC_F35_Khamis.png)

图 C.35 ACO 生成的最优路线

3. 下一个列表显示了使用 ACO 解决反渗透（RO）高压泵功率最大化问题的步骤。

列表 C.18 使用 ACO 解决 RO 高压泵功率最大化问题

```py
import random

md_range = (41.67, 416.67)                                            ①
nv_range = (1, 200)                                                   ①
delta_range = (1400, 2600)                                            ①
rr_range = (1, 50)                                                    ①
eta_range = (0.70, 0.85)                                              ①
rho = 1000                                                            ②

def ro_pump_power(X):                                                 ③
    md, nv, delta, rr, eta=X
    return (md ** 2 + 1200 * md * nv * delta) / (nv * rr * 3600 * eta *
➥ rho)

num_ants = 100                                                        ④
num_iterations = 300                                                  ④
evaporation_rate = 0.7                                                ④
pheromone_deposit = 1                                                 ④
initial_pheromone = 0.25                                              ④

pheromone_matrix = [[initial_pheromone] * 5 for _ in range(num_ants)] ⑤

best_solution = None                                                  ⑥
best_power = float('-inf')                                            ⑥

for _ in range(num_iterations):
    solutions = []
    powers = []

    for ant in range(num_ants):                                       ⑦
        md = random.uniform(md_range[0], md_range[1]) 
        nv = random.uniform(nv_range[0], nv_range[1]) 
        delta = random.uniform(delta_range[0], delta_range[1]) 
        rr = random.uniform(rr_range[0], rr_range[1]) 
        eta = random.uniform(eta_range[0], eta_range[1]) 

        soln=(md, nv, delta, rr, eta)                                 ⑧
        power = ro_pump_power(soln)                                   ⑧

        solutions.append((md, nv, delta, rr, eta))                    ⑨
        powers.append(power)                                          ⑨

        if power > best_power:                                        ⑩
            best_solution = (md, nv, delta, rr, eta)
            best_power = power

    for ant in range(num_ants):                                       ⑪
        for i in range(5):
            pheromone_matrix[ant][i] *= evaporation_rate
            if solutions[ant][i] == best_solution[i]:
                pheromone_matrix[ant][i] += pheromone_deposit / powers[ant]

print("Optimal Solution:")                                            ⑫
print("Md:", format(best_solution[0], '.2f'))                         ⑫
print("Nv:", format(best_solution[1], '.2f'))                         ⑫
print("Delta:", format(best_solution[2], '.2f'))                      ⑫
print("RR:", format(best_solution[3], '.2f'))                         ⑫
print("Eta:", format(best_solution[4], '.2f'))                        ⑫
print("Optimal HP:", format(best_power, '.2f'))                       ⑫
```

① 定义每个决策变量的范围。

② 水的密度为 kg/m³

③ 定义目标函数。

④ 定义 ACO 参数。

⑤ 初始化信息素矩阵。

⑥ 初始化最佳解决方案及其对应的功率。

⑦ 为每只蚂蚁构建解决方案。

⑧ 计算当前解决方案的功率。

⑨ 存储解决方案及其功率。

⑩ 如有必要，更新最佳解决方案。

⑪ 根据每个解决方案的强度更新信息素路径。

⑫ 打印决策变量的最优值和最优 HP。

以下是一个生成的输出示例：

```py
Optimal Solution:
Md: 404.10
Nv: 7.39
Delta: 2536.93
RR: 1.05
Eta: 0.76
Optimal HP: 425.75
```

作为 C.18 列表的延续，以下代码片段展示了使用 SciPy 中的`optimize`求解器解决此问题的步骤。

```py
from scipy.optimize import minimize                                    ①

def ro_pump_power(X):                                                  ②
    md, nv, delta, rr, eta=X
    return -(md ** 2 + 1200 * md * nv * delta) / (nv * rr * 3600 * eta *
➥ rho) 

bounds = [md_range, nv_range, delta_range, rr_range, eta_range]        ③
x0=[200, 100, 2000, 25, 0.75]                                          ④

result = minimize(ro_pump_power, x0, bounds=bounds, method='SLSQP')    ⑤

print("Optimal Solution:")                                             ⑥
print("Md:", format(result.x[0], '.2f'))                               ⑥
print("Nv:", format(result.x[1], '.2f'))                               ⑥
print("Delta:", format(result.x[2], '.2f'))                            ⑥
print("RR:", format(result.x[3], '.2f'))                               ⑥
print("Eta:", format(result.x[4], '.2f'))                              ⑥
print("Optimal HP:", format(-result.fun, '.2f'))                       ⑥
```

① 导入 scipy 优化器。

② 根据 scipy 的要求，定义带有负号的优化目标函数。

③ 设置决策变量的界限。

④ 设置初始猜测值。

⑤ 使用顺序最小二乘编程（SLSQP）解决优化问题。

⑥ 打印决策变量的最优值和最优 HP。

生成的输出如下：

```py
Optimal Solution:
Md: 416.67
Nv: 99.98
Delta: 2600.00
RR: 1.00
Eta: 0.70
Optimal HP: 515.88
```

您可以微调 ACO 参数以获得与 SciPy 优化器相当的结果。

4. 下一个列表展示了使用 MIDACO 求解器解决供需问题的步骤。我们首先导入库并设置问题数据。

列表 C.19 使用 MIDACO 解决供需问题

```py
import numpy as np
import random
import midaco                                                    ①
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import folium

supply_schools = [1, 3]                                          ②
demand_schools = [2, 4]                                          ③
amount_supply = [20, 30]                                         ④
amount_demand = [5, 45]                                          ⑤
n_var=len(supply_schools)*len(demand_schools)
n_constr=len(supply_schools)
```

① 导入 MIDACO 求解器。

② 设置有可用显微镜的学校。

③ 设置需要显微镜的学校。

④ 设置每所学校可用的显微镜数量。

⑤ 设置每所学校请求的显微镜数量。

作为延续，我们将为供应和需求学校生成随机的 GPS 坐标：
