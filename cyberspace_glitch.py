#!/usr/bin/env python3
"""
Cyberspace Glitch — Cyberpunk 2077-style cyberspace video effect pipeline.
SIGGRAPH 2021 技术方案的开源复现。
纯 CPU 实现，基于稀疏点云 + 故障特效。
"""

import os
import sys
import time
from collections import deque
import numpy as np
import cv2

# ── 强制 CPU 模式 ─────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
cv2.setNumThreads(1)

from rembg import remove, new_session
from PIL import Image

# ── 常量 ───────────────────────────────────────────────────────
TARGET_WIDTH = 1920           # 1080p 宽度（提高分辨率，让点阵颗粒更清晰）
NUM_POINTS = 8000             # 点云点数
Z_LAYERS = 3                  # 深度层数（3层）
PIXEL_SORT_STRIPS = 15        # 像素排序竖条数
DATAMOSH_RECTS = 3            # Datamoshing 矩形区域数（指令要求3个）
RECT_MIN_SIZE = 50
RECT_MAX_SIZE = 150
BLEND_CURRENT = 0.6           # 帧混合比例（指令要求）
BLEND_PREV = 0.3
BLEND_PREV2 = 0.1
BRIGHTNESS_THRESHOLD = 100    # 像素排序亮度阈值
SCANLINE_INTERVAL = 6         # 扫描线间隔
CHROMA_OFFSET = 3             # 色散偏移（RGB各偏移3像素）

# 人物残影参数
GHOST_MAX_FRAMES = 3          # 保留最近几帧的残影
GHOST_ALPHA_DECAY = 0.6       # 每帧透明度衰减系数
GHOST_INITIAL_ALPHA = 0.7     # 新残影初始透明度

# ── 鲜亮颜色方案（BGR） ──
BG_COLORS_BGR = [  # 蓝/青色系：大段维持高饱和蓝色，只在最高光才泛白
    (30, 10, 0), (70, 25, 0), (120, 50, 0), (170, 80, 0),
    (220, 120, 0), (255, 160, 0), (255, 190, 0), (255, 210, 10),
    (255, 225, 40), (255, 235, 90), (255, 245, 160), (255, 255, 255)
]

FG_COLORS_BGR = [  # 红色系：大段维持高饱和纯红，只在最高光才泛白
    (0, 0, 20), (0, 0, 60), (0, 0, 110), (0, 0, 160),
    (0, 0, 210), (0, 0, 255), (0, 0, 255), (0, 0, 255),
    (10, 10, 255), (60, 60, 255), (150, 150, 255), (255, 255, 255)
]


def build_color_lut(color_palette, size=256):
    """
    将离散的颜色列表通过线性插值转换为平滑的 256 级 LUT。
    返回 (size, 3) 的 BGR 数组。
    """
    if len(color_palette) < 2:
        return np.tile(np.array(color_palette[0], dtype=np.float32), (size, 1))
    orig_positions = np.linspace(0, size - 1, len(color_palette))
    target_positions = np.arange(size)
    palette = np.array(color_palette, dtype=np.float32)
    lut = np.empty((size, 3), dtype=np.float32)
    for ch in range(3):
        lut[:, ch] = np.interp(target_positions, orig_positions, palette[:, ch])
    return np.clip(lut, 0, 255).astype(np.uint8)


# 预计算平滑 LUT
BG_LUT = build_color_lut(BG_COLORS_BGR, 256)
BG_LUT = np.clip(BG_LUT.astype(np.float32) * 1.0, 0, 255).astype(np.uint8)
FG_LUT = build_color_lut(FG_COLORS_BGR, 256)
FG_LUT = np.clip(FG_LUT.astype(np.float32) * 1.25, 0, 255).astype(np.uint8)


def map_luminance_to_color(lum: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """将亮度值(0~255)通过 LUT 映射为 BGR 颜色。"""
    idx = np.clip(lum.astype(np.int32), 0, 255)
    return lut[idx]



# ── 点阵贴图缓存：把"点阵"预生成为可平铺的浮点蒙版，矢量化绘制而非逐点 cv2.circle ──
_DOT_MASK_CACHE = {}


def get_dot_mask(h, w, pitch_x=10, pitch_y=3, radius=1):
    """
    生成/缓存一张 h×w 的重复点阵蒙版（0~1）。
    pitch_y 小、pitch_x 大 → 竖直方向密集、水平方向疏松，呈现"网格点"质感。
    """
    key = (h, w, pitch_x, pitch_y, radius)
    if key in _DOT_MASK_CACHE:
        return _DOT_MASK_CACHE[key]
    tile = np.zeros((pitch_y, pitch_x), dtype=np.float32)
    cv2.circle(tile, (pitch_x // 2, pitch_y // 2), radius, 1.0, -1)
    reps_y = h // pitch_y + 2
    reps_x = w // pitch_x + 2
    big = np.tile(tile, (reps_y, reps_x))[:h, :w]
    _DOT_MASK_CACHE[key] = big
    return big


def generate_digital_wall(w, h, layers=2, seed=42):
    """
    生成背景点阵的深度层参数：每层一个整体亮度系数 + 轻微的水平采样偏移，
    用于在"内容采样"的基础上叠加一点层次感（近层更亮、偏移更大，制造纵深）。
    结构本身（点的位置）由 get_dot_mask 决定，这里只负责"深度感"的调制参数。
    """
    rng = np.random.default_rng(seed)
    layers_data = []
    for layer in range(layers):
        depth = (layer + 1) / layers          # 0.5 / 1.0，越大代表越"近"、越亮
        x_shift = int(rng.integers(-6, 7) * (layer + 1))
        layers_data.append({"depth": depth, "x_shift": x_shift})
    return layers_data


def draw_digital_wall(canvas, frame, layers_data, lut,
                      dot_pitch_x=20, dot_pitch_y=4, dot_radius=1,
                      brightness_floor=35, brightness_ceil=255):
    """
    根据原始帧内容采样绘制点阵背景：每个点的颜色由该位置在原始视频上的亮度决定
    （而不是纯程序化生成），经过 brightness_floor 兜底后映射到 BG_LUT，
    保证暗区也不会因为亮度过低而"隐形"（这是最早版本背景呈圆形的根因，这里明确规避）。

    点阵位置固定为网格（竖密横疏），pitch 与 dot_radius 的比例经过校正，
    保证相邻点之间有真实的黑色间隙、不会因为半径大于间距而糊成连续线条。
    """
    h, w = canvas.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dot_mask_full = get_dot_mask(h, w, dot_pitch_x, dot_pitch_y, dot_radius)
    canvas_f = np.zeros((h, w, 3), dtype=np.float32)
    span = brightness_ceil - brightness_floor

    for layer in layers_data:
        depth = layer["depth"]
        x_shift = layer["x_shift"]
        # 用小幅水平位移采样，制造多层视差叠加的纵深感（同一份内容，略微错位）
        sampled = np.roll(gray, x_shift, axis=1)
        brightness = np.clip(brightness_floor + span * (sampled / 255.0) * (0.5 + 0.5 * depth), 0, 255)
        color = map_luminance_to_color(brightness, lut).astype(np.float32)  # (h, w, 3)
        layer_rgb = dot_mask_full[:, :, None] * color
        canvas_f = np.maximum(canvas_f, layer_rgb)

    canvas[:] = np.clip(canvas_f, 0, 255).astype(np.uint8)
    return canvas



def generate_test_video(output_path: str, num_frames: int = 500,
                        width: int = 1280, height: int = 720) -> None:
    """生成彩色测试视频（当 input.mp4 不存在时使用）。"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    for i in range(num_frames):
        t = i / num_frames
        r = int(128 + 127 * np.sin(t * 6 + 0))
        g = int(128 + 127 * np.sin(t * 6 + 2))
        b = int(128 + 127 * np.sin(t * 6 + 4))

        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (b, g, r)

        cx = int(width * 0.3 + width * 0.4 * np.sin(t * 3))
        cy = int(height * 0.5 + height * 0.3 * np.cos(t * 2.5))
        cv2.circle(frame, (cx, cy), 120, (255, 255, 255), -1)
        cv2.circle(frame, (cx - 30, cy - 30), 20, (0, 0, 0), -1)
        cv2.circle(frame, (cx + 30, cy - 30), 20, (0, 0, 0), -1)
        cv2.ellipse(frame, (cx, cy + 40), (40, 20), 0, 0, 180, (0, 0, 0), 3)

        cv2.putText(frame, "CYBERSPACE TEST PATTERN",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame {i + 1}/{num_frames}",
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        out.write(frame)

    out.release()
    print(f"[TEST] Generated test video: {output_path} ({num_frames} frames)")


def compute_3d_noise(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """
    基于坐标的伪随机 3D 噪声，返回 [-1, 1] 之间的值。
    使用多重正弦波混合模拟低频空间噪声，无需外部库。
    """
    noise = (np.sin(xs * 0.1 + zs * 0.3) * np.cos(ys * 0.13 - zs * 0.2) +
             np.sin(ys * 0.07 + xs * 0.05 + zs * 0.4) * 0.6 +
             np.cos(xs * 0.03 - ys * 0.09 + zs * 0.15) * 0.3)
    noise /= 1.9  # 近似归一化至 [-1, 1]
    return noise.astype(np.float32)


def render_pointcloud(frame: np.ndarray, mask: np.ndarray,
                       num_points: int = NUM_POINTS, frame_idx: int = 0) -> np.ndarray:
    """
    核心人物渲染器：三层叠加，让人物是"实体数字化身"而不是稀疏噪点。
    frame: 原始帧 (BGR, uint8)
    mask:  rembg mask (单通道, uint8, 0=背景, 255=人物)
    返回人物画布。
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mask_f = mask.astype(np.float32) / 255.0

    # ── 1. 密集点阵基底：按亮度整体映射到 FG_LUT，用细点阵材质铺满人物轮廓 ──
    # （用 pitch=3 的高密度点阵，覆盖率远高于原来 8000 个稀疏矩形，才不会看起来像噪点）
    body_dots = get_dot_mask(h, w, pitch_x=3, pitch_y=3, radius=1) * mask_f
    body_colors = map_luminance_to_color(gray, FG_LUT).astype(np.float32)
    body_layer = body_colors * body_dots[:, :, np.newaxis]
    canvas = np.clip(body_layer, 0, 255).astype(np.uint8)

    # ── 2. 边缘加权稀疏高光点：保留五官/轮廓细节，制造故障闪烁感 ──
    edges = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    edges = np.abs(edges)
    edges = cv2.GaussianBlur(edges, (5, 5), 0)
    weight = edges * mask_f
    weight_sum = weight.sum()
    if weight_sum <= 0:
        weight = mask_f.copy()
        weight_sum = weight.sum() if weight.sum() > 0 else 1.0

    prob = weight.flatten() / weight_sum
    all_indices = np.arange(h * w)
    n_highlight = max(1, num_points // 3)  # 基底已经承担了"实体感"，高光点只做点缀
    person_points = np.random.choice(all_indices, size=n_highlight, p=prob, replace=True)

    point_ys = person_points // w
    point_xs = person_points % w

    z_layers = np.random.randint(0, Z_LAYERS, size=n_highlight)
    parallax_offsets = (z_layers + 1).astype(np.float32)
    point_xs_offset = (point_xs.astype(np.float32) + parallax_offsets).astype(np.int32)
    point_xs_offset = np.clip(point_xs_offset, 0, w - 1)
    point_ys = np.clip(point_ys, 0, h - 1)

    brightness = gray[point_ys, point_xs_offset]
    noise_vals = compute_3d_noise(point_xs.astype(np.float32), point_ys.astype(np.float32),
                                   z_layers.astype(np.float32))
    modulated_brightness = np.clip(brightness * (1.0 + noise_vals * 0.3), 0, 255)

    base_colors = map_luminance_to_color(modulated_brightness, FG_LUT)
    colors = np.clip(base_colors.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)  # 高光比基底更亮，才能"跳出来"

    for idx in range(n_highlight):
        px = int(point_xs_offset[idx])
        py = int(point_ys[idx])
        size = np.random.randint(2, 5)
        x1 = max(0, min(w - 1, px - size // 2))
        x2 = max(0, min(w - 1, px + size // 2))
        y1 = max(0, min(h - 1, py - size // 2))
        y2 = max(0, min(h - 1, py + size // 2))
        color = tuple(int(c) for c in colors[idx])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)

    # ── 3. 数字雨：几条随时间下落的竖直光流，仅在人物范围内绘制 ──
    canvas = draw_rain_streaks(canvas, mask, frame_idx)

    return canvas


def draw_rain_streaks(canvas: np.ndarray, mask: np.ndarray, frame_idx: int,
                      n_streaks: int = 16, seed: int = 7) -> np.ndarray:
    """在人物轮廓范围内绘制若干条随帧数下落的数字雨光流，呼应参考图里人物身上的竖直光效。"""
    h, w = canvas.shape[:2]
    rng = np.random.default_rng(seed)
    xs = rng.integers(0, w, size=n_streaks)
    speeds = rng.uniform(5, 12, size=n_streaks)
    lengths = rng.integers(30, 90, size=n_streaks)
    phases = rng.integers(0, h, size=n_streaks)

    for i in range(n_streaks):
        x = int(xs[i])
        if not (0 <= x < w):
            continue
        y_head = int((phases[i] + frame_idx * speeds[i]) % (h + lengths[i])) - lengths[i]
        y0, y1 = max(0, y_head), min(h, y_head + lengths[i])
        if y1 <= y0:
            continue
        col = mask[y0:y1, x]
        if not np.any(col > 128):
            continue  # 该列不经过人物，跳过（避免遮挡背景墙）
        for yy in range(y0, y1):
            if mask[yy, x] < 128:
                continue
            fade = 1.0 - abs((yy - y_head) / max(1, lengths[i]))
            b = int(np.clip(180 + 75 * fade, 0, 255))
            color = FG_LUT[b].astype(np.float32)
            canvas[yy, x] = np.clip(
                canvas[yy, x].astype(np.float32) * 0.4 + color * 0.9, 0, 255
            ).astype(np.uint8)
    return canvas


def pixel_sort(frame: np.ndarray, strips: int = PIXEL_SORT_STRIPS,
               threshold: int = BRIGHTNESS_THRESHOLD) -> np.ndarray:
    """
    像素排序：将画面垂直切为 strips 个竖条，
    在每个竖条内对亮度高于 threshold 的像素按亮度升序重排。
    """
    h, w = frame.shape[:2]
    result = frame.copy()
    strip_width = w // strips

    for s in range(strips):
        x_start = s * strip_width
        x_end = (s + 1) * strip_width if s < strips - 1 else w

        strip = result[:, x_start:x_end].copy()
        gray_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)

        # 对每一列处理
        for col in range(strip.shape[1]):
            column_data = strip[:, col, :].copy()
            col_gray = gray_strip[:, col]

            bright_mask = col_gray > threshold
            if np.sum(bright_mask) < 2:
                continue

            # 按亮度升序排序
            bright_indices = np.where(bright_mask)[0]
            bright_vals = column_data[bright_indices]
            brightness = col_gray[bright_indices]
            sorted_order = np.argsort(brightness)
            column_data[bright_indices] = bright_vals[sorted_order]

            strip[:, col, :] = column_data

        result[:, x_start:x_end] = strip

    return result


def datamosh(frame: np.ndarray, prev_frame: np.ndarray,
             num_rects: int = DATAMOSH_RECTS) -> np.ndarray:
    """
    数据混杂：从 prev_frame（上上帧）随机选择矩形区域，
    替换当前帧对应位置的像素，模拟 I-frame 损坏。
    """
    if prev_frame is None:
        return frame

    result = frame.copy()
    h, w = frame.shape[:2]

    for _ in range(num_rects):
        rw = np.random.randint(RECT_MIN_SIZE, RECT_MAX_SIZE + 1)
        rh = np.random.randint(RECT_MIN_SIZE, RECT_MAX_SIZE + 1)
        x = np.random.randint(0, max(1, w - rw))
        y = np.random.randint(0, max(1, h - rh))

        result[y:y + rh, x:x + rw] = prev_frame[y:y + rh, x:x + rw]

    return result


def chromatic_aberration(frame: np.ndarray, offset: int = CHROMA_OFFSET) -> np.ndarray:
    """
    色散效果：
    R 通道向右偏移 offset，
    G 通道向上偏移 offset，
    B 通道向左偏移 offset。
    """
    h, w = frame.shape[:2]
    result = frame.copy()

    # R 通道右移 offset
    r_shifted = np.roll(frame[:, :, 2], offset, axis=1)
    result[:, offset:, 2] = r_shifted[:, offset:]

    # G 通道上移 offset
    g_shifted = np.roll(frame[:, :, 1], -offset, axis=0)
    result[:h-offset, :, 1] = g_shifted[:h-offset, :]

    # B 通道左移 offset
    b_shifted = np.roll(frame[:, :, 0], -offset, axis=1)
    result[:, :w-offset, 0] = b_shifted[:, :w-offset]

    return result


def apply_scanlines(frame: np.ndarray, interval: int = SCANLINE_INTERVAL,
                    alpha: float = 0.15) -> np.ndarray:
    """扫描线效果：隔行插入半透明黑线。"""
    result = frame.copy().astype(np.float32)
    h = frame.shape[0]

    for y in range(0, h, interval):
        result[y, :, :] = result[y, :, :] * (1.0 - alpha)

    return np.clip(result, 0, 255).astype(np.uint8)


def frame_blend(current: np.ndarray, prev: np.ndarray,
                prev2: np.ndarray) -> np.ndarray:
    """残影拖尾：指数衰减帧混合（0.6/0.3/0.1）。"""
    result = current.astype(np.float32) * BLEND_CURRENT

    if prev is not None:
        result += prev.astype(np.float32) * BLEND_PREV
    if prev2 is not None:
        result += prev2.astype(np.float32) * BLEND_PREV2

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_bloom(frame: np.ndarray, threshold: int = 180,
                 blur_size: int = 21, intensity: float = 0.8,
                 max_bright_ratio: float = 0.12) -> np.ndarray:
    """提取高亮区域，高斯模糊后叠加回原图，模拟发光溢出。
    含过曝面积保护：高亮像素占比超过 max_bright_ratio 时，按比例衰减 intensity，
    避免大面积连续高亮内容（如密集点阵/条形）把画面糊成一片白。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    bright_ratio = float(np.count_nonzero(bright_mask)) / bright_mask.size

    effective_intensity = intensity
    if bright_ratio > max_bright_ratio:
        # 高亮面积越大，intensity 衰减越多，最低衰减到 0.25 倍
        overshoot = min(1.0, (bright_ratio - max_bright_ratio) / max_bright_ratio)
        effective_intensity = intensity * max(0.25, 1.0 - overshoot)

    bright_mask_f = bright_mask.astype(np.float32) / 255.0
    bright_areas = frame.astype(np.float32) * bright_mask_f[:, :, np.newaxis]

    glow = cv2.GaussianBlur(bright_areas, (blur_size, blur_size), 0)
    glow2 = cv2.GaussianBlur(bright_areas, (blur_size * 2 + 1, blur_size * 2 + 1), 0)

    result = frame.astype(np.float32) + (glow + glow2 * 0.5) * effective_intensity
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_contrast_boost(frame: np.ndarray, gamma: float = 0.85,
                         black_point: int = 10) -> np.ndarray:
    """
    压暗死黑、拉高中高亮部分的对比度曲线。
    参考图里红/蓝之所以"跳"，很大程度是背景够黑——这一步专门解决"发闷"的问题。
    """
    f = frame.astype(np.float32)
    f = np.clip(f - black_point, 0, 255) / max(1, (255 - black_point)) * 255
    f = 255 * (f / 255) ** gamma
    return np.clip(f, 0, 255).astype(np.uint8)


def glitch_intensity(frame_idx: int, base=0.15, burst_prob=0.03) -> float:
    """大部分时间保持低强度，偶尔（概率 burst_prob）触发强烈爆发。"""
    rng = np.random.default_rng(frame_idx)  # 确定性随机数
    if rng.random() < burst_prob:
        return float(rng.uniform(0.7, 1.0))   # 强烈故障
    return base + float(rng.uniform(0, 0.1))  # 平时的轻微抖动


def apply_glitch_effects(frame: np.ndarray, prev_frame: np.ndarray,
                          prev2_frame: np.ndarray, frame_idx: int = 0) -> np.ndarray:
    """
    后处理特效（动态故障强度）：
      像素排序 → 数据混杂 → 色散 → 扫描线 → 帧混合 → Bloom → 对比度提升
    """
    intensity = glitch_intensity(frame_idx)

    # 1. 像素排序
    sorted_frame = pixel_sort(frame)

    # 2. 数据混杂（数量随强度变化）
    num_rects = int(1 + 4 * intensity)
    moshed = datamosh(sorted_frame, prev2_frame if prev2_frame is not None else prev_frame,
                      num_rects=num_rects)

    # 3. 色散（动态偏移）
    chroma_offset = max(1, int(CHROMA_OFFSET * intensity * 2))
    chroma = chromatic_aberration(moshed, offset=chroma_offset)

    # 4. 扫描线
    scanned = apply_scanlines(chroma)

    # 5. 帧混合
    blended = frame_blend(scanned, prev_frame, prev2_frame)

    # 6. Bloom 发光
    blended = apply_bloom(blended)

    # 7. 对比度/伽马提升——压死黑、拉高光，避免整体发闷
    blended = apply_contrast_boost(blended)

    return blended


def main():
    # ── 路径解析 ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "input.mp4")
    output_path = os.path.join(script_dir, "output.mp4")

    # ── 检查/生成输入视频 ──
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open {input_path}, generating test video...")
        cap.release()
        generate_test_video(input_path, num_frames=500)
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("[ERROR] Failed to generate test video. Exiting.")
            sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0:
        print("[WARN] Empty video (0 frames). Generating test video...")
        cap.release()
        generate_test_video(input_path, num_frames=500)
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── 计算 720p 缩放尺寸 ──
    scale = TARGET_WIDTH / orig_width
    target_h = int(orig_height * scale)
    target_h = target_h if target_h > 0 else 720

    print(f"[INFO] Input: {orig_width}x{orig_height} @ {fps:.2f} fps, "
          f"{total_frames} frames")
    print(f"[INFO] Target: {TARGET_WIDTH}x{target_h} @ {fps:.2f} fps")
    print(f"[INFO] Config: points={NUM_POINTS}, z_layers={Z_LAYERS}, "
          f"pixel_sort_strips={PIXEL_SORT_STRIPS}, datamosh_rects={DATAMOSH_RECTS}")

    # ── 初始化 VideoWriter ──
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps,
                             (TARGET_WIDTH, target_h))
    if not writer.isOpened():
        print("[ERROR] Cannot open VideoWriter. Check codec support.")
        cap.release()
        sys.exit(1)

    # ── 初始化 rembg 会话（只加载一次） ──
    print("[INFO] Loading rembg u2net session (CPU)...")
    try:
        session = new_session("u2net", providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"[ERROR] Failed to create rembg session: {e}")
        print("[WARN] Falling back — will use a dummy mask (all-white).")
        session = None
    print("[INFO] rembg session ready.")

    # ── 帧缓冲区 ──
    prev_frame = None
    prev2_frame = None

    # ── 残影历史 ──
    ghost_frames = deque(maxlen=GHOST_MAX_FRAMES)

    # ── 背景数据墙的深度层参数（只需生成一次，颜色本身逐帧根据当前帧采样） ──
    layers_data = generate_digital_wall(TARGET_WIDTH, target_h)
    start_time = time.time()
    frame_idx = 0

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        frame_start = time.time()

        # ── 缩放到 1080p ──
        frame = cv2.resize(raw_frame, (TARGET_WIDTH, target_h))
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── rembg 抠图获取 mask ──
        try:
            if session is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_in = Image.fromarray(frame_rgb)
                pil_out = remove(pil_in, session=session,
                                 only_mask=False, post_process_mask=True)
                if isinstance(pil_out, Image.Image):
                    if pil_out.mode == "RGBA":
                        mask_pil = pil_out.getchannel("A")
                    else:
                        mask_pil = Image.fromarray(
                            np.full((target_h, TARGET_WIDTH), 255, dtype=np.uint8))
                else:
                    mask_pil = Image.fromarray(
                        np.full((target_h, TARGET_WIDTH), 255, dtype=np.uint8))
                mask = np.array(mask_pil)
            else:
                mask = np.full((target_h, TARGET_WIDTH), 255, dtype=np.uint8)
        except Exception as e:
            print(f"[WARN] rembg error on frame {frame_idx}: {e}, using full mask")
            mask = np.full((target_h, TARGET_WIDTH), 255, dtype=np.uint8)

        # ── 背景：逐帧根据当前帧内容采样绘制（颜色跟随视频背景变化），
        #     再挖掉人物轮廓区域，避免人物点阵间隙里透出背景的蓝色点，把红色弄脏 ──
        bg_canvas = np.zeros((target_h, TARGET_WIDTH, 3), dtype=np.uint8)
        draw_digital_wall(bg_canvas, frame, layers_data, BG_LUT)
        bg_canvas[mask > 128] = 0

        # ── 渲染人物点云 ──
        person_canvas = render_pointcloud(frame, mask, num_points=NUM_POINTS, frame_idx=frame_idx)

        # ── 人物残影 ──
        ghost_canvas = np.zeros_like(person_canvas, dtype=np.float32)
        for i, hcanvas in enumerate(ghost_frames):
            # 距今越近的帧权重越高
            weight = GHOST_INITIAL_ALPHA * (GHOST_ALPHA_DECAY ** (len(ghost_frames) - 1 - i))
            ghost_canvas += hcanvas.astype(np.float32) * weight
        ghost_canvas = np.clip(ghost_canvas, 0, 255).astype(np.uint8)
        ghost_frames.append(person_canvas.copy())  # 保存当前人物点云供后续帧使用

        # ── 合并点云（背景 + 残影 + 当前人物） ──
        combined = cv2.add(bg_canvas, ghost_canvas)
        combined = cv2.add(combined, person_canvas)

        # ── 后处理特效 ──
        output_frame = apply_glitch_effects(combined, prev_frame, prev2_frame, frame_idx)

        # ── 写入输出 ──
        writer.write(output_frame)

        # ── 帧缓冲区滚动 ──
        if prev2_frame is not None:
            del prev2_frame
        prev2_frame = prev_frame
        prev_frame = output_frame.copy()

        # ── 释放内存 ──
        del mask, bg_canvas, person_canvas, combined

        # ── 进度打印 ──
        elapsed = time.time() - frame_start
        total_elapsed = time.time() - start_time
        eta = (total_elapsed / frame_idx) * (total_frames - frame_idx) if frame_idx > 0 else 0

        progress_pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
        print(f"[{frame_idx}/{total_frames}] {progress_pct:.1f}%  "
              f"frame_time={elapsed:.2f}s  ETA={eta:.0f}s")

        if elapsed > 5.0:
            print("    [INFO] 处理中，请勿关闭...")

    # ── 清理 ──
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    total_time = time.time() - start_time
    print(f"[DONE] Output saved to: {output_path}")
    print(f"[DONE] Processed {frame_idx} frames in {total_time:.1f}s "
          f"({frame_idx / total_time:.2f} fps avg)")


if __name__ == "__main__":
    main()