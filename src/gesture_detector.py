"""
gesture_detector.py — Motor de reconhecimento de gestos
Inspirado em: detection_gesture_app.py (cognitive ref) + algoritmos PRD seção 6
Padrão: funções puras isoladas + identify_gesture() como único orquestrador
"""

import math

import config
from src.utils import (
    euclidean_distance,
    get_palm_center,
    normalize_hand_size,
)


# ───────── AJUSTE ADAPTATIVO POR TAMANHO DA MÃO ─────────

def _adaptive_scale(hand_landmarks) -> float:
    """
    Escala adaptativa de thresholds baseada no tamanho relativo da mão.
    Clampa para evitar sensibilidade extrema em casos de oclusão/parcial.
    """
    hand_size = normalize_hand_size(hand_landmarks)
    if hand_size <= 1e-6:
        return 1.0

    reference_size = 0.12
    scale = hand_size / reference_size
    return min(1.4, max(0.75, scale))


# ───────── ESTADO DE VOLANTE (ÂNGULO 2 MÃOS) ─────────

_steering_neutral_angle = None
_steering_delta_ema = 0.0
_steering_last_direction = "NEUTRO"

STEERING_PROFILES = {
    "SUAVE": {
        "deadzone_deg": 11.0,
        "activation_deg": 16.0,
        "smoothing_alpha": 0.18,
    },
    "PADRAO": {
        "deadzone_deg": config.STEERING_DEADZONE_DEG,
        "activation_deg": config.STEERING_ACTIVATION_DEG,
        "smoothing_alpha": config.STEERING_SMOOTHING_ALPHA,
    },
    "COMPETITIVO": {
        "deadzone_deg": 6.0,
        "activation_deg": 9.0,
        "smoothing_alpha": 0.35,
    },
}


def _normalize_angle_deg(angle: float) -> float:
    """Normaliza ângulo para intervalo [-180, 180]."""
    return ((angle + 180.0) % 360.0) - 180.0


def reset_steering_state():
    """Reseta estado interno do volante (útil para testes/calibração)."""
    global _steering_neutral_angle, _steering_delta_ema, _steering_last_direction
    _steering_neutral_angle = None
    _steering_delta_ema = 0.0
    _steering_last_direction = "NEUTRO"


def _get_steering_profile_params() -> dict:
    """Retorna parâmetros do perfil de volante configurado."""
    profile = str(getattr(config, "STEERING_PROFILE", "PADRAO")).upper()
    return STEERING_PROFILES.get(profile, STEERING_PROFILES["PADRAO"])


# ───────── DETECÇÃO DE MÃO FECHADA (PUNHO) ─────────

def is_closed_hand(hand_landmarks) -> bool:
    """
    Verifica se a mão está fechada (punho).
    Critério: pontas de todos os dedos próximas ao centro da palma.
    """
    palm  = get_palm_center(hand_landmarks)
    lm    = hand_landmarks.landmark
    scale = _adaptive_scale(hand_landmarks)
    closed_threshold = config.CLOSED_HAND_THRESHOLD * scale

    tips = [lm[4], lm[8], lm[12], lm[16], lm[20]]
    # Acessibilidade: requer apenas 3 de 5 dedos (maioria)
    closed_count = sum(1 for tip in tips if euclidean_distance(tip, palm) < closed_threshold)
    return closed_count >= 3


# ───────── DETECÇÃO DE MÃO ABERTA (PALMA) ─────────

def is_open_hand(hand_landmarks) -> bool:
    """
    Verifica se a mão está aberta (dedos estendidos).
    Critério: pontas de todos os dedos distantes do centro da palma.
    """
    palm = get_palm_center(hand_landmarks)
    lm   = hand_landmarks.landmark
    scale = _adaptive_scale(hand_landmarks)
    open_threshold = config.OPEN_HAND_THRESHOLD * scale

    tips = [lm[4], lm[8], lm[12], lm[16], lm[20]]
    # Acessibilidade: requer apenas 3 de 5 dedos (maioria)
    open_count = sum(1 for tip in tips if euclidean_distance(tip, palm) > open_threshold)
    return open_count >= 3


# ───────── DETECÇÃO DO GESTO V (VICTORY) ─────────

def is_victory_sign(hand_landmarks) -> bool:
    """
    Detecta o sinal V (indicador + médio abertos, anelar + mínimo fechados).
    Inspirado em is_victory() do cognitive ref.
    """
    palm = get_palm_center(hand_landmarks)
    lm   = hand_landmarks.landmark
    scale = _adaptive_scale(hand_landmarks)
    v_open_threshold = config.V_OPEN_THRESHOLD * scale
    v_closed_threshold = config.V_CLOSED_THRESHOLD * scale

    index_open  = euclidean_distance(lm[8],  palm) > v_open_threshold
    middle_open = euclidean_distance(lm[12], palm) > v_open_threshold
    ring_closed = euclidean_distance(lm[16], palm) < v_closed_threshold
    pinky_closed= euclidean_distance(lm[20], palm) < v_closed_threshold

    return index_open and middle_open and ring_closed and pinky_closed


def is_thumbs_up(hand_landmarks) -> bool:
    """
    Detecta o gesto de 'Joinha' (polegar para cima, outros dedos dobrados).
    Muito mais estável para toggles do que a pinça.
    """
    lm = hand_landmarks.landmark
    palm = get_palm_center(hand_landmarks)
    scale = _adaptive_scale(hand_landmarks)
    
    # Polegar deve estar acima da palma e distante
    thumb_tip = lm[4]
    is_thumb_up = thumb_tip.y < lm[3].y and thumb_tip.y < lm[2].y
    
    # Outros dedos devem estar dobrados (próximos à palma)
    tips = [lm[8], lm[12], lm[16], lm[20]]
    closed_threshold = config.CLOSED_HAND_THRESHOLD * scale
    
    # Acessibilidade: Requer que a maioria (3 de 4) esteja fechada
    closed_count = sum(1 for tip in tips if euclidean_distance(tip, palm) < closed_threshold)
    
    return is_thumb_up and closed_count >= 3


# ───────── DETECÇÃO DE PALMAS JUNTAS (PAUSAR) ─────────

def are_palms_together(left_hand, right_hand) -> bool:
    """
    Detecta se as duas palmas estão juntas (pausa o jogo).
    Critério: distância entre centros das palmas < threshold.
    """
    left_palm  = get_palm_center(left_hand)
    right_palm = get_palm_center(right_hand)
    left_scale = _adaptive_scale(left_hand)
    right_scale = _adaptive_scale(right_hand)
    threshold = config.PALMS_TOGETHER_THRESHOLD * ((left_scale + right_scale) / 2)
    return euclidean_distance(left_palm, right_palm) < threshold


# ───────── DETECÇÃO DE DIREÇÃO DO VOLANTE ─────────

def get_steering_direction(left_hand, right_hand) -> str:
    """
    Detecta movimento de volante pelo ângulo de inclinação entre as duas palmas.
    Retorna: 'ESQUERDA', 'DIREITA' ou 'NEUTRO'
    """
    global _steering_neutral_angle, _steering_delta_ema, _steering_last_direction

    left_palm = get_palm_center(left_hand)
    right_palm = get_palm_center(right_hand)

    dx = right_palm.x - left_palm.x
    dy = right_palm.y - left_palm.y
    span = math.hypot(dx, dy)

    if span < config.STEERING_MIN_SPAN:
        _steering_last_direction = "NEUTRO"
        return {"direction": "NEUTRO", "angle": 0.0}

    raw_angle = math.degrees(math.atan2(dy, dx))

    if _steering_neutral_angle is None:
        _steering_neutral_angle = raw_angle

    raw_delta = _normalize_angle_deg(raw_angle - _steering_neutral_angle)
    profile_params = _get_steering_profile_params()
    alpha = profile_params["smoothing_alpha"]
    _steering_delta_ema = ((1.0 - alpha) * _steering_delta_ema) + (alpha * raw_delta)

    # Se estiver realmente perto do neutro, adapta lentamente o ponto neutro para postura natural
    if abs(raw_delta) <= profile_params["deadzone_deg"]:
        drift = _normalize_angle_deg(raw_angle - _steering_neutral_angle)
        _steering_neutral_angle = _normalize_angle_deg(
            _steering_neutral_angle + (config.STEERING_NEUTRAL_ADAPT * drift)
        )

    enter = profile_params["activation_deg"]
    exit_zone = profile_params["deadzone_deg"]

    # Histerese: para manter estabilidade, sair da curva exige retornar à zona morta
    direction = "NEUTRO"
    if _steering_last_direction == "ESQUERDA":
        if _steering_delta_ema <= -exit_zone:
            direction = "ESQUERDA"
        else:
            _steering_last_direction = "NEUTRO"
    elif _steering_last_direction == "DIREITA":
        if _steering_delta_ema >= exit_zone:
            direction = "DIREITA"
        else:
            _steering_last_direction = "NEUTRO"
    elif _steering_delta_ema <= -enter:
        _steering_last_direction = "ESQUERDA"
        direction = "ESQUERDA"
    elif _steering_delta_ema >= enter:
        _steering_last_direction = "DIREITA"
        direction = "DIREITA"

    return {"direction": direction, "angle": _steering_delta_ema}


# ───────── ORQUESTRADOR PRINCIPAL ─────────

def identify_gesture(left_hand, right_hand) -> dict:
    """
    Identifica o estado dos controles com suporte analógico.
    """
    state = {
        "steering": "NEUTRO",
        "steering_angle": 0.0,
        "hand_height": 1.0,
        "pedal":    "NEUTRO",
        "action":   "NEUTRO"
    }

    if left_hand is None and right_hand is None:
        return state

    # 1. AÇÕES (Prioridade Máxima)
    if left_hand is not None and right_hand is not None and are_palms_together(left_hand, right_hand):
        state["action"] = "PAUSAR"
        return state

    # 2. VOLANTE ANALÓGICO
    if left_hand is not None and right_hand is not None:
        steering_res = get_steering_direction(left_hand, right_hand)
        state["steering"] = steering_res["direction"]
        state["steering_angle"] = steering_res["angle"]
        
        # Altura média para acelerador analógico
        l_palm = get_palm_center(left_hand)
        r_palm = get_palm_center(right_hand)
        state["hand_height"] = (l_palm.y + r_palm.y) / 2.0

    # ✌️ ITEM / TROCAR (Somente se não estiver fazendo pausa)
    if state["action"] == "NEUTRO":
        left_is_v = left_hand is not None and is_victory_sign(left_hand)
        right_is_v = right_hand is not None and is_victory_sign(right_hand)
        if left_is_v and not right_is_v:
            state["action"] = "ITEM"
        elif right_is_v and not left_is_v:
            state["action"] = "TROCAR"

    # 3. FREIO (Mãos abertas - Prioridade sobre aceleração no Mapper)
    if left_hand is not None and right_hand is not None:
        if is_open_hand(left_hand) and is_open_hand(right_hand):
            state["pedal"] = "FREAR"

    return state
