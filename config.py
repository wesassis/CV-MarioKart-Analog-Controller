# ───────── CONFIGURAÇÕES GERAIS DO PROJETO ─────────

# ───────── CÂMERA E VÍDEO ─────────
CAMERA_INDEX     = 0
FRAME_WIDTH      = 640
FRAME_HEIGHT     = 480
PREVIEW_MIRRORED = True  # Ativa o espelhamento para corrigir a inversão natural
DRAW_HAND_LANDMARKS = True 

# ───────── RASTREAMENTO (MEDIAPIPE) ─────────
MAX_NUM_HANDS            = 2
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE  = 0.5

# ───────── TECLAS E MAPEAMENTO ─────────
from pynput.keyboard import Key

KEY_ACCEL  = 'a'
KEY_BRAKE  = Key.down
KEY_LEFT   = 'a'
KEY_RIGHT  = 'd'
KEY_ITEM   = Key.space
KEY_SWITCH = 'z'
KEY_PAUSE  = Key.esc

# Mapeamento para o KeyboardMapper (Fallback)
GESTURE_KEY_MAP = {
    "ACELERAR": KEY_ACCEL,
    "FREAR":    KEY_BRAKE,
    "ESQUERDA": KEY_LEFT,
    "DIREITA":  KEY_RIGHT,
    "ITEM":     KEY_ITEM,
    "TROCAR":   KEY_SWITCH,
    "PAUSAR":   KEY_PAUSE,
    "TOGGLE":   None,
}

# ───────── THRESHOLDS DE GESTOS ─────────
CLOSED_HAND_THRESHOLD    = 0.10
OPEN_HAND_THRESHOLD      = 0.14
V_OPEN_THRESHOLD         = 0.16
V_CLOSED_THRESHOLD       = 0.08
PINCH_THRESHOLD          = 0.05
PALMS_TOGETHER_THRESHOLD = 0.08

# ───────── VOLANTE (ÂNGULO E SENSIBILIDADE) ─────────
STEERING_DEADZONE_DEG    = 8.0
STEERING_ACTIVATION_DEG  = 12.0
STEERING_SMOOTHING_ALPHA = 0.25
STEERING_NEUTRAL_ADAPT   = 0.04
STEERING_PROFILE         = "PADRAO"
STEERING_MIN_SPAN        = 0.15
STEERING_THRESHOLD       = 0.15 # Usado para desenhar linhas no HUD

# ───────── PROCESSAMENTO E TIMING ─────────
TARGET_FPS     = 30
CONFIRM_FRAMES = 2
PAUSE_HOLD_MS  = 800

# Servidor SSE (Frontend Monitor)
SSE_HOST = "localhost"
SSE_PORT = 8765

# Intervalos de Debounce (Segundos)
GESTURE_INTERVALS = {
    "ESQUERDA":  0.03,
    "DIREITA":   0.03,
    "ITEM":      0.80,
    "TROCAR":    0.50,
    "TOGGLE":    1.50,
    "PAUSAR":    1.50,
    "NEUTRO":    0.00,
}

# ───────── HUD / DESIGN VISUAL ─────────
HUD_COLOR_NEUTRAL = (180, 180, 180)
HUD_COLOR_ITEM    = (50, 255, 50)
HUD_COLOR_LOOK    = (255, 50, 255)
HUD_COLOR_PAUSE   = (50, 50, 255)
HUD_COLOR_LEFT    = (50, 200, 255)
HUD_COLOR_RIGHT   = (50, 200, 255)
HUD_COLOR_FPS     = (0, 255, 0)

# Mapeamento de Cores para o HUD
GESTURE_COLOR_MAP = {
    "ESQUERDA": HUD_COLOR_LEFT,
    "DIREITA":  HUD_COLOR_RIGHT,
    "ITEM":     HUD_COLOR_ITEM,
    "TROCAR":   HUD_COLOR_LOOK,
    "PAUSAR":   HUD_COLOR_PAUSE,
    "NEUTRO":   HUD_COLOR_NEUTRAL,
}

# ───────── MODO DE CALIBRAÇÃO ─────────
CALIBRATION_MODE = False  # Define como True para desativar aceleração temporariamente
