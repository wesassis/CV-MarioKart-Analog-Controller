"""
utils.py — Funções matemáticas puras para cálculo de landmarks
Inspirado em: helpers.py (VR Tracking) + math.hypot do body.py (mamuskas)
"""

import math
import numpy as np


# ───────── DISTÂNCIA ─────────

def euclidean_distance(p1, p2) -> float:
    """Distância euclidiana entre dois landmarks normalizados (x, y)"""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def euclidean_distance_xy(x1, y1, x2, y2) -> float:
    """Distância euclidiana entre dois pontos (x, y) escalares"""
    return math.hypot(x1 - x2, y1 - y2)


# ───────── CENTROS DE MÃO ─────────

def get_palm_center(hand_landmarks):
    """
    Centro da palma calculado pela média dos landmarks base.
    Pontos: 0 (pulso), 1, 5, 9, 13, 17 (bases dos dedos)
    """
    palm_indices = [0, 1, 5, 9, 13, 17]
    landmarks = hand_landmarks.landmark
    xs = [landmarks[i].x for i in palm_indices]
    ys = [landmarks[i].y for i in palm_indices]

    class _Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    return _Point(sum(xs) / len(xs), sum(ys) / len(ys))


def get_hand_center_y(hand_landmarks) -> float:
    """Centro Y da mão (média de todos os landmarks) para detecção de volante"""
    landmarks = hand_landmarks.landmark
    return sum(lm.y for lm in landmarks) / len(landmarks)


def get_hand_center_x(hand_landmarks) -> float:
    """Centro X da mão (média de todos os landmarks) para suavização"""
    landmarks = hand_landmarks.landmark
    return sum(lm.x for lm in landmarks) / len(landmarks)


# ───────── TAMANHO RELATIVO DA MÃO ─────────

def normalize_hand_size(hand_landmarks) -> float:
    """
    Tamanho relativo da mão baseado na distância pulso → base do dedo médio.
    Usado para thresholds adaptativos por usuário.
    Retorna valor normalizado (distância / comprimento de referência).
    """
    lm = hand_landmarks.landmark
    wrist  = lm[0]   # Pulso
    middle = lm[9]   # Base do dedo médio
    return euclidean_distance(wrist, middle)


# ───────── POSIÇÃO DOS DEDOS ─────────

def get_finger_tip(hand_landmarks, finger_idx: int):
    """Retorna o landmark da ponta de um dedo pelo índice"""
    return hand_landmarks.landmark[finger_idx]


def get_finger_positions_px(img, hand_landmarks) -> list:
    """
    Retorna lista de (id, x_px, y_px) para todos os landmarks.
    Mesmo padrão do get_finger_positions() do cognitive ref.
    """
    h, w, _ = img.shape
    return [
        (idx, int(lm.x * w), int(lm.y * h))
        for idx, lm in enumerate(hand_landmarks.landmark)
    ]


# ───────── SUAVIZAÇÃO ─────────

def smooth_value(history) -> float:
    """Média dos valores no buffer de histórico (deque)"""
    if not history:
        return 0.5
    return sum(history) / len(history)
