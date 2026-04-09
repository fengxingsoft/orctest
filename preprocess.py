#!/usr/bin/env python3
"""
文档/OCR 预处理：面向“词汇表拍照图 -> 接近扫描件”

输出两张图：
1) OCR输入图（灰度增强）
2) 扫描件图（纯黑白）
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    # normalize_lighting
    target_mean: int = 180  # 建议 170~190
    clip_gain_min: float = 0.85
    clip_gain_max: float = 1.20

    # 轻量去噪（CLAHE前）
    pre_denoise_enabled: bool = True
    pre_denoise_median_ksize: int = 3  # 文档推荐 3

    # CLAHE（受控）
    clahe_clip_limit: float = 1.8  # 必须 <= 2.0
    clahe_tile_grid_size: int = 8

    # 自适应阈值（防断字）
    adaptive_block_size: int = 25  # 奇数，建议 21~35
    adaptive_c: int = 4  # 建议 <= 5

    # 连通域过滤（面积 + 宽高）
    min_area: int = 10
    w_thresh: int = 4
    h_thresh: int = 4
    
    # 彩色批注抑制（针对红笔/彩笔）
    suppress_colored_marks: bool = True
    chroma_thresh: int = 28
    dark_keep_thresh: int = 110

    # 可选轻微锐化（默认关，必要时开）
    sharpen_enabled: bool = False
    sharpen_alpha: float = 1.20


def ensure_odd(v: int, min_odd: int = 3) -> int:
    if v < min_odd:
        v = min_odd
    if v % 2 == 0:
        v += 1
    return v


def normalize_lighting(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """温和亮度归一化：避免把背景噪点过度拉亮。"""
    mean_val = float(np.mean(gray))
    if mean_val < 1e-6:
        return gray

    gain = cfg.target_mean / mean_val
    gain = float(np.clip(gain, cfg.clip_gain_min, cfg.clip_gain_max))

    corrected = np.clip(gray.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return corrected


def remove_shadow(gray: np.ndarray) -> np.ndarray:
    """背景估计法去阴影，保持文字结构。"""
    # 大尺度形态学闭运算估计慢变化背景
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # 用差分去除阴影，再归一化
    diff = cv2.absdiff(background, gray)
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    return norm


def controlled_clahe(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """受控CLAHE，限制噪点放大。"""
    tile = max(4, cfg.clahe_tile_grid_size)
    clahe = cv2.createCLAHE(
        clipLimit=min(cfg.clahe_clip_limit, 2.0),
        tileGridSize=(tile, tile),
    )
    return clahe.apply(gray)


def optional_sharpen(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """轻微锐化（可选）。"""
    if not cfg.sharpen_enabled:
        return gray

    # 注意：仅用 very mild kernel，避免笔画断裂
    alpha = max(1.0, min(cfg.sharpen_alpha, 1.5))
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharp = cv2.filter2D(gray, -1, kernel)
    out = cv2.addWeighted(sharp, alpha - 1.0, gray, 2.0 - alpha, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def auto_tune_for_vocab_sheet(gray: np.ndarray, cfg: PreprocessConfig) -> None:
    """简单自动调参：仅做小幅度自适应，避免破坏原结构。"""
    std = float(np.std(gray))

    # 对比度偏低：略增CLAHE；对比度高：更保守
    if std < 35:
        cfg.clahe_clip_limit = min(2.0, max(1.6, cfg.clahe_clip_limit + 0.2))
        cfg.adaptive_block_size = 27
        cfg.adaptive_c = min(cfg.adaptive_c, 4)
    elif std > 60:
        cfg.clahe_clip_limit = min(cfg.clahe_clip_limit, 1.6)
        cfg.adaptive_block_size = 23
        cfg.adaptive_c = min(cfg.adaptive_c, 3)

    cfg.adaptive_block_size = ensure_odd(cfg.adaptive_block_size)


def remove_small_cc_noise(binary_inv: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """
    连通域过滤（黑底白字输入）。
    删除规则（必须）：
      if area < min_area AND (width < w_thresh OR height < h_thresh): 删除
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_inv, connectivity=8)

    out = np.zeros_like(binary_inv)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        should_remove = (area < cfg.min_area) and (w < cfg.w_thresh or h < cfg.h_thresh)
        if not should_remove:
            out[labels == i] = 255

    return out


def build_neutral_ink_mask(image_bgr: np.ndarray, gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """
    生成“接近黑/灰墨迹”掩膜，用于抑制彩色批注。
    原理：彩色笔通常通道差异大（高chroma），黑灰字通道差异小。
    """
    b = image_bgr[:, :, 0].astype(np.int16)
    g = image_bgr[:, :, 1].astype(np.int16)
    r = image_bgr[:, :, 2].astype(np.int16)

    chroma = np.abs(r - g) + np.abs(g - b) + np.abs(r - b)

    # 深色字即便带轻微色偏也保留，避免误伤细笔画
    neutral = chroma < cfg.chroma_thresh
    dark = gray < cfg.dark_keep_thresh
    keep_mask = np.logical_or(neutral, dark)
    return (keep_mask.astype(np.uint8) * 255)


def binarize_for_scan(gray: np.ndarray, image_bgr: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    block_size = ensure_odd(cfg.adaptive_block_size)
    c_val = min(cfg.adaptive_c, 5)

    bw_inv = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=block_size,
        C=c_val,
    )
    
    if cfg.suppress_colored_marks:
        neutral_ink_mask = build_neutral_ink_mask(image_bgr, gray, cfg)
        bw_inv = cv2.bitwise_and(bw_inv, neutral_ink_mask)

    cleaned_inv = remove_small_cc_noise(bw_inv, cfg)

    # 保护表格线与细笔画：仅做 very mild close，避免变细/断裂
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned_inv = cv2.morphologyEx(cleaned_inv, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 反色成“白底黑字”
    scan_bw = 255 - cleaned_inv
    return scan_bw


def build_ocr_input(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    # 去阴影
    no_shadow = remove_shadow(gray)

    # 关键：CLAHE之前先做轻量去噪
    if cfg.pre_denoise_enabled:
        k = ensure_odd(cfg.pre_denoise_median_ksize)
        no_shadow = cv2.medianBlur(no_shadow, k)

    # 自动参数微调（可关闭，这里默认开启）
    auto_tune_for_vocab_sheet(no_shadow, cfg)

    # 受控对比增强
    enhanced = controlled_clahe(no_shadow, cfg)

    # 温和亮度归一化（避免噪点放大）
    normalized = normalize_lighting(enhanced, cfg)

    # 可选轻微锐化
    ocr_gray = optional_sharpen(normalized, cfg)

    return ocr_gray


def preprocess_document(image_bgr: np.ndarray, cfg: PreprocessConfig) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    ocr_gray = build_ocr_input(gray, cfg)
    scan_bw = binarize_for_scan(ocr_gray, image_bgr, cfg)

    return ocr_gray, scan_bw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="词汇表OCR预处理（输出OCR灰度图 + 扫描件黑白图）")
    parser.add_argument("input", type=str, help="输入图片路径")
    parser.add_argument("--out-dir", type=str, default=".", help="输出目录")

    # 常用参数暴露
    parser.add_argument("--target-mean", type=int, default=180)
    parser.add_argument("--clahe-clip", type=float, default=1.8)
    parser.add_argument("--block-size", type=int, default=25)
    parser.add_argument("--c", type=int, default=4)
    parser.add_argument("--min-area", type=int, default=10)
    parser.add_argument("--w-thresh", type=int, default=4)
    parser.add_argument("--h-thresh", type=int, default=4)
    parser.add_argument("--chroma-thresh", type=int, default=28, help="彩色抑制阈值（越小越严格）")
    parser.add_argument("--dark-keep-thresh", type=int, default=110, help="深色保留阈值（防误删细字）")
    parser.add_argument("--no-color-suppress", action="store_true", help="关闭彩色批注抑制")
    parser.add_argument("--sharpen", action="store_true", help="启用轻微锐化")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(f"无法读取输入图像: {args.input}")

    cfg = PreprocessConfig(
        target_mean=args.target_mean,
        clahe_clip_limit=min(args.clahe_clip, 2.0),
        adaptive_block_size=args.block_size,
        adaptive_c=min(args.c, 5),
        min_area=args.min_area,
        w_thresh=args.w_thresh,
        h_thresh=args.h_thresh,
        suppress_colored_marks=not args.no_color_suppress,
        chroma_thresh=args.chroma_thresh,
        dark_keep_thresh=args.dark_keep_thresh,
        sharpen_enabled=args.sharpen,
    )

    ocr_gray, scan_bw = preprocess_document(img, cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(args.input).stem
    ocr_path = out_dir / f"{stem}_ocr_input.png"
    scan_path = out_dir / f"{stem}_scan_bw.png"

    cv2.imwrite(str(ocr_path), ocr_gray)
    cv2.imwrite(str(scan_path), scan_bw)

    print(f"[OK] OCR输入图: {ocr_path}")
    print(f"[OK] 扫描件图: {scan_path}")


if __name__ == "__main__":
    main()
