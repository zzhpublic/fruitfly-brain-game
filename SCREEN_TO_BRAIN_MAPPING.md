# 游戏画面/现实世界到果蝇大脑系统的映射文档

## 概述

本文档详细描述了现实世界图像或游戏画面如何通过视网膜式编码映射到果蝇大脑神经网络系统，形成完整的"感知-决策-动作"闭环。

---

## 完整映射流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        现实世界 / 游戏画面                                     │
│                         (RGB 帧, 如 160×120)                                  │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GameScreenProcessor.process_frame()                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ 1. 灰度化: cv2.cvtColor(frame, COLOR_RGB2GRAY)                          │ │
│  │ 2. 降采样: cv2.resize(gray, (64, 64))  → 8×8 受容野网格                  │ │
│  │ 3. 编码方式选择 (4种):                                                   │ │
│  │    ├─ frame_diff     : 帧差分 |gray_t - gray_{t-1}| → 运动检测           │ │
│  │    ├─ optical_flow   : Farneback 光流 → 运动矢量幅度                     │ │
│  │    ├─ intensity      : 直接像素强度 → 静态物体检测                       │ │
│  │    └─ edges          : Canny 边缘检测 → 物体边界                         │ │
│  │ 4. 网格池化: _pool_to_grid() 将 64×64 池化为 8×8=64 个神经元             │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ spike_rates: (64,) Hz, 范围 0-100
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  _map_screen_to_input() 电流注入映射                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ INPUT_SCALE = 5000.0  # 关键缩放因子                                     │ │
│  │                                                                         │ │
│  │ for i, rate in enumerate(spike_rates):                                 │ │
│  │     if rate > 0:                                                        │ │
│  │         idx = int(i * scale) % n_optic_neurons                         │ │
│  │         I_optic[idx] += rate * INPUT_SCALE  # pA                        │ │
│  │                                                                         │ │
│  │ # 结果: external_input["optic_lobes"] = I_optic  (形状: n_optic,)      │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ 外部电流 (pA)
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NetworkBuilder.step(I_ext) 网络步进                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ 1. LIFPopulation.step(I_ext) 更新膜电位                                 │ │
│  │    - I_syn_ext 参数接收外部电流                                          │ │
│  │    - V += dt/C_m * (-g_L*(V-E_L) + I_syn + I_ext)                      │ │
│  │    - 阈值 -40mV, 漏电 -60mV, 需 ~500,000 pA 触发脉冲                    │ │
│  │ 2. 突触传播: prev_spikes → 突触电流 → 后突触神经元                       │ │
│  │    - 视叶 → 中央复合体 → 蘑菇体 → 外侧角 → 下行神经元                    │ │
│  │ 3. 返回 all_spikes: Dict[region, spike_indices]                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ 脉冲数据
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    _decode_action() 动作解码                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ # Pinball 示例:                                                         │ │
│  │ left_spikes  = action[:5].sum()   # 5个左侧下行神经元                    │ │
│  │ right_spikes = action[5:].sum()   # 5个右侧下行神经元                    │ │
│  │                                                                         │ │
│  │ left_rate  = left_spikes  / 5.0                                         │ │
│  │ right_rate = right_spikes / 5.0                                         │ │
│  │                                                                         │ │
│  │ move = (right_rate - left_rate) * 15.0  # 像素/步                       │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ 动作向量
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         游戏环境 step(action)                                 │
│  - 更新球/板位置                                                             │
│  - 计算奖励                                                                  │
│  - 返回新观测                                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 两套编码体系对比

### 1. 通用屏幕处理器 (demo_closed_loop.py)

适用于**任意游戏画面**，无需游戏内部状态。

```python
class GameScreenProcessor:
    def __init__(self, n_neurons=64, grid_size=(8,8), method="frame_diff"):
        self.method = method  # 4种编码方式
        
    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (grid_size[0]*8, grid_size[1]*8))
        
        if method == "frame_diff":
            return self._process_frame_diff(gray)
        elif method == "optical_flow":
            return self._process_optical_flow(gray)
        # ... intensity, edges
        
    def _pool_to_grid(self, img, scale=1.0):
        # 将图像池化为 n_neurons 个神经元的脉冲率
        cell_h, cell_w = img.shape[0]//grid_y, img.shape[1]//grid_x
        for i, j in grid:
            spike_rates[idx] = img[y1:y2, x1:x2].mean() * scale
        return spike_rates
```

**特点**：
- 纯视觉输入，不依赖游戏内部状态
- 类似视网膜→外侧膝状体→初级视觉皮层的早期视觉处理
- 适合迁移学习、跨游戏泛化

### 2. 游戏专用编码 (src/games/*.py)

每个游戏环境内置**生物学更合理**的编码，利用游戏内部状态。

#### Pinball 编码 (`src/games/pinball.py:_encode_observation`)

```python
# 64个运动检测神经元 (8×8 受容野)
for i, (cx, cy) in enumerate(receptive_fields):
    dist = sqrt((cx - ball_x)² + (cy - ball_y)²)
    pos_rate = exp(-dist² / 800) * 50      # 高斯位置调谐
    motion_rate = (abs(dx) + abs(dy)) * 10  # 运动敏感度
    motion_rates[i] = min(pos_rate + motion_rate, 100)

# 32个位置编码神经元 (One-hot 位置编码)
ball_x_bin = int(ball_x / width * 16)   # 16 neurons
ball_y_bin = int(ball_y / height * 8)   # 8 neurons  
paddle_x_bin = int(paddle_x / width * 8) # 8 neurons

# 泊松脉冲生成
spike_prob = 1 - exp(-rate * obs_window / 1000)
spikes = random() < spike_prob
```

**生物学对应**：
- 运动检测神经元 ↔ **T4/T5 细胞** (方向选择性)
- 位置编码神经元 ↔ **视叶柱状结构** 的视野拓扑映射
- 泊松脉冲 ↔ **皮层神经元的不规则放电**

---

## 5个游戏的输入映射差异

| 游戏 | 视觉输入维度 | 编码方式 | 额外输入目标区域 | 任务类型 | 对应果蝇行为 |
|------|-------------|----------|-----------------|----------|-------------|
| **Pinball** | 96 (64运+32位) | 受容野+One-hot | - | 视觉运动控制 | 追踪移动目标 |
| **Pong** | 类似 | 受容野+One-hot | - | 视觉运动控制 | 拦截飞行物体 |
| **Maze** | 迷宫渲染帧 | 通用处理器 | 中央复合体(指南针) | 空间导航 | 寻找食物/巢穴 |
| **Odor** | 气味浓度场 | 梯度编码 | 蘑菇体(嗅觉) | 嗅觉导航 | 顺风寻找气味源 |
| **Looming** | 膨胀圆盘 | 边缘/膨胀检测 | - | 逃避反应 | 逃避捕食者/碰撞 |

---

## 关键参数详解

### INPUT_SCALE = 5000.0 (demo_closed_loop.py:305)

```python
# 为什么是 5000？
# LIF 神经元参数: V_th=-40mV, E_L=-60mV, C_m=200pF, g_L=10nS
# 静息态: V ≈ -60mV
# 触发脉冲需要: ΔV = 20mV
# 所需电荷: Q = C_m * ΔV = 200pF * 20mV = 4000 pC
# 在 dt=0.1ms 内: I = Q/dt = 4000pC / 0.1ms = 40,000 pA
# 但有漏电流: I_leak = g_L * (V - E_L) = 10nS * 20mV = 200 pA
# 实际需要更大电流克服漏电 + 突触整合
# 经验值: ~500,000 pA 外部电流可靠触发脉冲
# 
# spike_rate 最大 100 Hz → 100 * 5000 = 500,000 pA ✓
```

### conn_scale = 100 (config/connectome_test.yaml)

```yaml
simulation:
  conn_scale: 100  # 突触权重全局缩放
```

- 合成连接组初始权重 ~1.0 nS
- 缩放后 ~100 nS
- 单个突触电流: g * (E_rev - V) = 100nS * 60mV = 6000 pA
- 视叶→中央复合体 ~40 个连接 → 240,000 pA
- 配合循环放大足以触发下游脉冲

### obs_window = 50ms (pinball.py)

- 脉冲生成的时间窗口
- 对应果蝇视觉系统的积分时间常数
- 50ms ≈ 20 Hz 采样率，匹配果蝇视觉闪烁融合频率

---

## 代码关键路径

```
demo_closed_loop.py
├── GameScreenProcessor.process_frame()      # 通用视觉编码 (第 27-100 行)
├── ClosedLoopDemo._map_screen_to_input()    # 电流注入映射 (第 298-340 行)
├── ClosedLoopDemo.run()                     # 主循环 (第 400-550 行)
│   ├── frame = env.render()                 # 获取游戏画面
│   ├── spike_rates = processor.process_frame(frame)
│   ├── external_input = self._map_screen_to_input(spike_rates)
│   ├── spikes = network.step(external_input)
│   ├── action = self._decode_action(spikes)
│   └── obs, reward, done, _ = env.step(action)

src/games/pinball.py
├── PinballEnv._encode_observation()         # 专用编码 (第 100-145 行)
├── PinballEnv._decode_action()              # 动作解码 (第 158-170 行)
└── PinballEnv.step()                        # 环境步进 (第 172-280 行)

src/networks/assembly.py
├── NetworkBuilder.step()                    # 网络步进 (第 149-220 行)
│   ├── populations[region].step(I_ext)      # LIF 更新
│   └── 突触传播使用 prev_spikes             # 关键修复!
└── LIFPopulation.step()                     # 向量化 LIF (neurons/lif.py:95-180)
```

---

## 现实世界摄像头接入 (demo_camera.py)

```python
# 同理，摄像头帧直接送入 GameScreenProcessor
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    spike_rates = processor.process_frame(frame)
    external_input = demo._map_screen_to_input(spike_rates)
    spikes = network.step(external_input)
    # ... 解码动作控制机器人/无人机
```

**注意事项**：
1. **分辨率匹配**: 摄像头建议 160×120 或 320×240，避免过大导致处理延迟
2. **光照归一化**: 实际环境需加自动增益/直方图均衡
3. **延迟预算**: 捕获→编码→网络→解码→执行 < 100ms 为实时
4. **背景减除**: 静态背景用 `frame_diff` 或 `edges` 更鲁棒

---

## 扩展建议

### 1. 更生物学的视网膜模型
```python
# 可替换为:
# - DoG (Difference of Gaussian) 中心-周围受容野
# - 视网膜神经节细胞模型 (ON/OFF 型)
# - 事件相机 (DVS) 事件流直接输入
```

### 2. 多模态融合
```python
# 视觉 + 本体感觉 + 前庭觉
external_input = {
    "optic_lobes": I_visual,
    "central_complex": I_compass,      # 方向积分
    "mushroom_body": I_odor,           # 嗅觉
    "descending_neurons": I_proprio    # 本体感觉
}
```

### 3. 注意力机制
```python
# 自上而下的注意力调制视叶增益
# 蘑菇体/中央复合体 → 视叶反馈投射
I_optic *= attention_gain  # 1.0-3.0
```

---

## 总结

| 层级 | 对应生物结构 | 代码实现 | 关键参数 |
|------|-------------|----------|----------|
| **感受器** | 视网膜光感受器 | 游戏渲染/摄像头 | 分辨率、帧率 |
| **早期视觉** | 薄层/髓状体 (Lamina/Medulla) | GameScreenProcessor | 受容野大小、编码方式 |
| **运动检测** | T4/T5 细胞、叶状体 | frame_diff / optical_flow | 时间窗口 50ms |
| **特征整合** | 叶瓣/小叶瓣 | _pool_to_grid 池化 | 8×8 网格 |
| **下行命令** | 下行神经元 (DN) | _decode_action | 5左+5右神经元 |
| **运动执行** | 腿/翅运动神经元 | env.step(action) | 像素/步位移 |

这个映射完整实现了**视觉-运动变换**，是果蝇大脑闭环控制的核心通路。