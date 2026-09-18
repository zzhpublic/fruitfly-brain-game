# 下行神经元解码机制详细分析

## 概述

本文档详细分析果蝇大脑神经网络系统中下行神经元的解码机制，包括两层解码架构、各游戏的解码差异、生物学对应关系及改进建议。

---

## 两层解码架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        网络输出脉冲 (spikes dict)                              │
│  {                                                                          │
│    "optic_lobes": array([...]),                                             │
│    "central_complex": array([...]),                                         │
│    "mushroom_body": array([...]),                                           │
│    "lateral_horn": array([...]),                                            │
│    "descending_neurons": array([...])                                       │
│  }                                                                          │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  第1层: demo_closed_loop.py:_map_spikes_to_action()                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ 通用动作向量生成器                                                       │ │
│  │                                                                         │ │
│  │ 不同游戏读取不同脑区:                                                    │ │
│  │   - pong/maze/looming → central_complex (导航决策)                      │ │
│  │   - odor → lateral_horn (嗅觉价性/本能行为)                              │ │
│  │   - pinball → descending_neurons (直接运动命令)                          │ │
│  │                                                                         │ │
│  │ 统一输出: np.ndarray (动作向量)                                          │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ action: np.ndarray
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  第2层: src/games/*.py:_decode_action()                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ 游戏专用物理动作解码                                                     │ │
│  │                                                                         │ │
│  │ Pinball:  差分编码 → 水平位移 (-15~15 px)                                │ │
│  │ Pong:     差分编码 → 垂直位移 (-10~10 px)                                │ │
│  │ Maze:     差分+求和 → 转角速度+前进速度                                  │ │
│  │ Odor:     同 Maze (增益更低)                                             │ │
│  │ Looming:  种群投票 → 布尔逃跑决策                                        │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ 物理动作
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    env.step(action) 执行动作                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 各游戏下行神经元解码详细对比

### 1. Pinball (弹球) - 最完整的实现

**读取脑区**: `descending_neurons` (直接运动输出层)

**动作神经元分配**: 15 个神经元分 3 组
```
索引 0-4   : 左转群体 (5 个神经元)
索引 5-9   : 保持群体 (5 个神经元) 
索引 10-14 : 右转群体 (5 个神经元)
```

**第1层解码** (`demo_closed_loop.py:420-435`):
```python
desc_spikes = spikes.get("descending_neurons", np.array([]))

left_spikes  = desc_spikes[:5].sum()    # 左群体总脉冲数
stay_spikes  = desc_spikes[5:10].sum()  # 中群体总脉冲数
right_spikes = desc_spikes[10:15].sum() # 右群体总脉冲数

# 率编码归一化 (脉冲数/神经元数)
rates = [left_spikes/5, stay_spikes/5, right_spikes/5]
action = np.array(rates)  # shape (3,)
```

**第2层解码** (`pinball.py:133-145`):
```python
def _decode_action(self, action: np.ndarray) -> float:
    left_spikes  = action[:5].sum()   # 注意: 这里 action 已经是 rates
    right_spikes = action[5:].sum()
    
    left_rate  = left_spikes / 5.0
    right_rate = right_spikes / 5.0
    
    # 差分编码: 右 - 左 → 正=右移, 负=左移
    move = (right_rate - left_rate) * 15.0  # 像素/步
    return move
```

**设计亮点**:
- ✅ **对抗性群体编码**: 左/右神经元竞争，差分解码得方向
- ✅ **率编码**: 脉冲计数归一化 → 连续速度值
- ✅ **增益匹配**: `*15.0` 像素/步，球速 4.0，反应足够快
- ⚠️ **保持群体未使用**: 第2层只用了左右，中间5个神经元被忽略

---

### 2. Pong (乒乓球)

**读取脑区**: `central_complex` (导航决策区)

**动作神经元**: 10 个 (5上 + 5下)

**解码** (`pong.py:97-107`):
```python
def _decode_action(self, action: np.ndarray) -> float:
    up_spikes = action[:5].sum()
    down_spikes = action[5:].sum()
    
    up_rate = up_spikes / 5.0
    down_rate = down_spikes / 5.0
    
    move = (up_rate - down_rate) * 10.0  # 垂直位移
    return move
```

---

### 3. Maze (迷宫导航)

**读取脑区**: `central_complex` (空间定向、路径整合)

**动作神经元**: 12 个 (4左转 + 4右转 + 4前进)

**解码** (`maze.py:152-162`):
```python
def _decode_action(self, action: np.ndarray) -> Tuple[float, float]:
    turn_left = action[:4].sum() / 4.0
    turn_right = action[4:8].sum() / 4.0
    forward = action[8:].sum() / 4.0
    
    turn_rate = (turn_right - turn_left) * 0.5  # rad/step
    speed = forward * 0.5  # units/step
    
    return turn_rate, speed
```

---

### 4. Odor (嗅觉导航)

**读取脑区**: `lateral_horn` (嗅觉价性、本能行为)

**动作神经元**: 12 个 (同 Maze)

**解码** (`odor.py:112-120`):
```python
def _decode_action(self, action: np.ndarray) -> Tuple[float, float]:
    turn_left = action[:4].sum() / 4.0
    turn_right = action[4:8].sum() / 4.0
    forward = action[8:].sum() / 4.0
    
    turn_rate = (turn_right - turn_left) * 0.3  # 增益更低
    speed = forward * 0.3
    
    return turn_rate, speed
```

**差异**: 增益 0.3 vs 0.5，嗅觉导航更平滑、更慢

---

### 5. Looming (迫近逃避)

**读取脑区**: `central_complex` (巨大纤维通路上游)

**动作神经元**: 10 个

**解码** (`looming.py:91-96`):
```python
def _decode_action(self, action: np.ndarray) -> bool:
    """种群投票逃跑决策"""
    escape_votes = action.sum()
    return escape_votes > 5 * 25  # 阈值: 125 总脉冲数
```

**特点**: 简单阈值投票，输出布尔值 (逃跑/不逃跑)

---

## 解码参数汇总表

| 游戏 | 读取脑区 | 神经元数 | 分组 | 解码方式 | 输出 | 增益 |
|------|---------|---------|------|---------|------|------|
| Pinball | descending_neurons | 15 | 5/5/5 | 差分率编码 | 标量位移 | 15.0 |
| Pong | central_complex | 10 | 5/5 | 差分率编码 | 标量位移 | 10.0 |
| Maze | central_complex | 12 | 4/4/4 | 差分+求和 | (角速度, 速度) | 0.5 |
| Odor | lateral_horn | 12 | 4/4/4 | 差分+求和 | (角速度, 速度) | 0.3 |
| Looming | central_complex | 10 | - | 种群投票 | 布尔 | - |

---

## 生物学对应关系

| 代码概念 | 果蝇解剖学结构 | 功能描述 | 文献支持 |
|---------|--------------|---------|---------|
| `descending_neurons` | DNa, DNb, DNc, DNd, DNg, DNp, MDN | 运动命令从脑下行到腹神经索 | Namiki et al. 2018; Card 2012 |
| `central_complex` | PB(协议桥), EB(椭圆体), FB(扇形体), NO(结节) | 空间定向、路径整合、导航决策 | Seelig & Jayaraman 2015; Green et al. 2017 |
| `lateral_horn` | LH (外侧角) | 嗅觉价性处理、本能行为触发 | Dolan et al. 2019; Frechter et al. 2019 |
| 左/右对抗群体 | DN 左/右侧投射神经元 | 双侧运动控制、转向决策 | Bidaye et al. 2020 |
| 差分解码 | 互抑制回路 (GABA能) | Winner-take-all 决策 | Tuthill & Wilson 2016 |

---

## 关键问题分析

### 问题 1: 读取脑区不一致
```python
# Pinball 读取 descending_neurons (正确: 直接运动输出)
# Pong/Maze/Looming 读取 central_complex (间接: 导航决策→运动)
# Odor 读取 lateral_horn (嗅觉价性→运动)
```
**影响**: 训练时奖励信号回传路径不同，学习动态不一致

### 问题 2: Pinball 保持群体浪费
```python
# 第1层生成 3 组 rates [left, stay, right]
# 第2层只用 left 和 right，stay 被丢弃
```

### 问题 3: Looming 过于简单
```python
# 简单阈值投票，无渐变逃跑强度
# 生物学上: 巨大纤维 GF → TTMn → 跳跃，有延迟和强度编码
```

### 问题 4: 硬编码维度
```python
# 新增游戏需同时改:
# 1. demo_closed_loop.py:_map_spikes_to_action() 
# 2. src/games/new_game.py:_decode_action()
```

---

## 改进建议

### 1. 统一动作空间定义
```python
# src/games/action_spaces.py
from enum import Enum
from dataclasses import dataclass

class ActionSpaceType(Enum):
    CONTINUOUS_1D = "continuous_1d"    # Pinball, Pong
    CONTINUOUS_2D = "continuous_2d"    # Maze, Odor
    DISCRETE_BINARY = "discrete_binary" # Looming

@dataclass
class ActionSpaceSpec:
    space_type: ActionSpaceType
    n_neurons: int
    readout_region: str
    neuron_groups: dict  # {"left": (0,5), "right": (5,10), ...}
    decode_fn: callable
```

### 2. 游戏环境声明式接口
```python
# src/games/pinball.py
class PinballEnv(gym.Env):
    # 声明动作空间规格
    ACTION_SPEC = ActionSpaceSpec(
        space_type=ActionSpaceType.CONTINUOUS_1D,
        n_neurons=10,  # 5左+5右 (去掉stay)
        readout_region="descending_neurons",
        neuron_groups={"left": (0,5), "right": (5,10)},
        decode_fn=decode_continuous_1d_diff
    )
```

### 3. 通用解码器库
```python
# src/games/decoders.py
def decode_continuous_1d_diff(rates: np.ndarray, groups: dict, gain: float) -> float:
    """1D 连续动作: 差分编码"""
    left = rates[groups["left"][0]:groups["left"][1]].mean()
    right = rates[groups["right"][0]:groups["right"][1]].mean()
    return (right - left) * gain

def decode_continuous_2d(rates: np.ndarray, groups: dict, gains: tuple) -> tuple:
    """2D 连续动作: 转向+前进"""
    turn = (rates[groups["right"]].mean() - rates[groups["left"]].mean()) * gains[0]
    speed = rates[groups["forward"]].mean() * gains[1]
    return turn, speed

def decode_discrete_vote(rates: np.ndarray, threshold: float) -> bool:
    """离散投票"""
    return rates.sum() > threshold

def decode_with_integration(spikes: np.ndarray, tau: float = 20.0) -> np.ndarray:
    """时间积分解码: 模拟下行神经元膜电位积分"""
    # 低通滤波器: rate_t = rate_{t-1} * exp(-dt/tau) + spikes_t
    pass

def population_vector_decode(left_rates: np.ndarray, right_rates: np.ndarray) -> tuple:
    """种群向量解码"""
    # 向量和法: angle = atan2(sum(right), sum(left))
    pass
```

### 4. 重构 demo_closed_loop.py
```python
def _map_spikes_to_action(self, spikes: dict) -> np.ndarray:
    spec = self.env.ACTION_SPEC
    region_spikes = spikes.get(spec.readout_region, np.array([]))
    
    if len(region_spikes) == 0:
        return np.zeros(spec.n_neurons)
    
    # 将脉冲索引转为率 (假设每步 dt=0.1ms, 窗口 50ms)
    rates = np.zeros(spec.n_neurons)
    for i in range(spec.n_neurons):
        idx = int(i * len(region_spikes) / spec.n_neurons)
        if idx < len(region_spikes):
            rates[i] = region_spikes[idx] * 10.0  # 脉冲→率
    
    return spec.decode_fn(rates, spec.neuron_groups, spec.decode_gains)
```

---

## 代码位置速查

| 文件 | 函数/类 | 行号 | 说明 |
|------|---------|------|------|
| `demo_closed_loop.py` | `_map_spikes_to_action` | 392 | 第1层通用解码 |
| `src/games/pinball.py` | `_decode_action` | 133 | Pinball 第2层解码 |
| `src/games/pong.py` | `_decode_action` | 97 | Pong 第2层解码 |
| `src/games/maze.py` | `_decode_action` | 152 | Maze 第2层解码 |
| `src/games/odor.py` | `_decode_action` | 112 | Odor 第2层解码 |
| `src/games/looming.py` | `_decode_action` | 91 | Looming 第2层解码 |

---

## 总结

当前系统实现了**基于率编码的对抗性群体解码**，基本符合果蝇下行运动控制的生物学原理：

1. **核心机制**: 左/右神经元群体竞争 → 差分解码 → 运动方向
2. **优势**: 简单、鲁棒、生物学合理
3. **不足**: 硬编码多、脑区读取不一致、缺乏时间积分、Looming 过简化

**优先改进**:
1. 统一 `readout_region` 为 `descending_neurons` (所有游戏)
2. 提取通用解码器库，消除重复代码
3. 加入时间积分机制，模拟真实下行神经元动力学
4. 为 Looming 添加渐变逃跑强度输出