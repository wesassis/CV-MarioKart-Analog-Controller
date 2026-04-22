"""
keyboard_mapper.py — State machine + controle de teclado
Inspirado em: Controller class atual + GESTURE_INTERVALS do cognitive ref
Evolução: adiciona N-frames confirm + state machine completa + hold/tap por gesto
"""

import time
from collections import deque
from pynput.keyboard import Key, Controller as KeyboardController

import config


# ───────── GESTOS QUE FICAM PRESSIONADOS (hold) vs DISPARADOS (tap) ─────────

HOLD_GESTURES = {"ACELERAR", "FREAR", "ESQUERDA", "DIREITA"}
TAP_GESTURES  = {"ITEM", "TROCAR", "PAUSAR", "TOGGLE"}

# Grupos para evitar que um comando solte outro de categoria diferente
GESTURE_GROUPS = {
    "ACELERAR": "PEDAL",
    "FREAR":    "PEDAL",
    "ESQUERDA": "STEERING",
    "DIREITA":  "STEERING"
}

PHASE_IDLE = "IDLE"
PHASE_HAND_DETECTED = "HAND_DETECTED"
PHASE_GESTURE_RECOGNIZED = "GESTURE_RECOGNIZED"
PHASE_COMMAND_FIRED = "COMMAND_FIRED"
PHASE_DEBOUNCE = "DEBOUNCE"


class KeyboardMapper:
    """
    Mapeia gestos detectados em eventos de teclado.
    Gerencia state machine: IDLE → DETECTED → CONFIRMED → FIRING → DEBOUNCE → IDLE
    Mantém padrão da Controller class atual (press/release/tap/release_all).
    """

    def __init__(self):
        self.kb      = KeyboardController()
        self.pressed = set()                          # Teclas atualmente pressionadas

        # ── State machine ──
        self._gesture_state     = "NEUTRO"            # Estado atual
        self._frame_buffer      = deque(maxlen=config.CONFIRM_FRAMES)  # Buffer N-frames
        self._last_gesture_times = {}                 # Debounce por gesto (igual cognitive ref)
        self._phase = PHASE_IDLE
        self._pause_hold_start = None
        self.accel_enabled = True # Começa ligado (mockado)
        self._toggle_lock = False # Trava para evitar disparos repetidos (flicker)

    # ───────── API PÚBLICA ─────────

    def update(self, state: dict):
        """
        Versão Ultra-Simples: Toggle por Pinça + Steering Contínuo.
        """
        action = state["action"]
        
        # ─── 1. AUTO-TOGGLE POR ALTURA (ACESSIBILIDADE) ───
        # Y < 0.70 significa mãos acima da linha de 70% da tela (de cima para baixo)
        height = state.get("hand_height", 1.0)
        self.accel_enabled = height < 0.70
        
        # ─── 2. OUTRAS AÇÕES (ITEM, TROCAR, PAUSA) ───
        if action in {"ITEM", "TROCAR", "PAUSAR"}:
            if action == "PAUSAR":
                action = self._apply_pause_hold_guard(action)
            if action in TAP_GESTURES:
                self._apply_tap(action)

        # ─── 3. VOLANTE (HOLD CONTÍNUO) ───
        steering = state["steering"]
        if steering == "ESQUERDA":
            self.press(config.KEY_LEFT)
            self.release(config.KEY_RIGHT)
        elif steering == "DIREITA":
            self.press(config.KEY_RIGHT)
            self.release(config.KEY_LEFT)
        else:
            self.release(config.KEY_LEFT)
            self.release(config.KEY_RIGHT)

        # ─── 4. FREIO ───
        pedal = state["pedal"]
        if pedal == "FREAR":
            self.press(config.KEY_BRAKE)
        else:
            self.release(config.KEY_BRAKE)

        # ─── 5. AUTO-ACCELERATE (MODO TOGGLE) ───
        self._ensure_auto_accelerate(action if action != "NEUTRO" else pedal)

        # Estado para o HUD
        self._gesture_state = action if action != "NEUTRO" else (
            pedal if pedal != "NEUTRO" else steering
        )

    def get_state(self) -> str:
        """Retorna o gesto principal confirmado para o HUD"""
        return self._gesture_state

    def get_phase(self) -> str:
        """Retorna a fase atual da state machine"""
        return self._phase

    def is_accel_on(self) -> bool:
        """Retorna se a aceleração está ligada no toggle"""
        return self.accel_enabled

    # ───────── CONTROLE DE TECLA (mesmo padrão do Controller atual) ─────────

    def press(self, key):
        """Pressiona uma tecla e registra no set de pressionadas"""
        if key not in self.pressed:
            self.kb.press(key)
            self.pressed.add(key)

    def release(self, key):
        """Solta uma tecla e remove do set de pressionadas"""
        if key in self.pressed:
            self.kb.release(key)
            self.pressed.discard(key)

    def tap(self, key):
        """Pressiona e solta uma tecla instantaneamente"""
        self.kb.press(key)
        time.sleep(0.05)
        self.kb.release(key)

    def release_all(self):
        """Solta todas as teclas pressionadas (cleanup)"""
        for k in list(self.pressed):
            self.release(k)

    # ───────── INTERNOS ─────────

    def _confirm_gesture(self) -> str:
        """
        Retorna o gesto somente se aparecer em todos os frames do buffer.
        Evita falsos positivos (inspirado em 'gesto confirmado após N frames' do PRD).
        """
        if len(self._frame_buffer) < config.CONFIRM_FRAMES:
            return "NEUTRO"

        # Todos os frames do buffer devem ser o mesmo gesto
        if len(set(self._frame_buffer)) == 1:
            return self._frame_buffer[0]

        return "NEUTRO"

    def _apply_pause_hold_guard(self, gesture: str) -> str:
        """Exige sustentação do gesto PAUSAR por um tempo mínimo."""
        if gesture != "PAUSAR":
            self._pause_hold_start = None
            return gesture

        now = time.time()
        if self._pause_hold_start is None:
            self._pause_hold_start = now
            return "NEUTRO"

        hold_ms = (now - self._pause_hold_start) * 1000.0
        if hold_ms < config.PAUSE_HOLD_MS:
            return "NEUTRO"

        return "PAUSAR"

    def _should_fire(self, gesture: str) -> bool:
        """
        Verifica se o gesto passou do intervalo de debounce.
        Mesmo padrão de last_gesture_times do cognitive ref.
        """
        now      = time.time()
        interval = config.GESTURE_INTERVALS.get(gesture, 0.1)
        last     = self._last_gesture_times.get(gesture, 0)

        if now - last >= interval:
            self._last_gesture_times[gesture] = now
            self._phase = PHASE_COMMAND_FIRED
            return True
        self._phase = PHASE_DEBOUNCE
        return False

    def _apply_hold(self, gesture: str):
        """Aplica gesto de hold: mantém a tecla pressionada"""
        key = config.GESTURE_KEY_MAP.get(gesture)
        if key is None:
            return

        # Solta teclas APENAS do mesmo grupo (ex: Esquerda solta Direita, mas não solta Acelerar)
        current_group = GESTURE_GROUPS.get(gesture)
        for g, k in config.GESTURE_KEY_MAP.items():
            if g in HOLD_GESTURES and g != gesture:
                if GESTURE_GROUPS.get(g) == current_group:
                    self.release(k)

        if self._should_fire(gesture):
            self.press(key)

    def _apply_tap(self, gesture: str):
        """Aplica gesto de tap: pressiona e solta uma vez"""
        key = config.GESTURE_KEY_MAP.get(gesture)
        if key is None:
            return

        # Taps não devem soltar comandos de hold nesta nova arquitetura multi-estado
        if self._should_fire(gesture):
            self.tap(key)

    def _ensure_auto_accelerate(self, gesture: str):
        """Aceleração constante se o toggle estiver ligado."""
        accel_key = config.KEY_ACCEL

        # Se o toggle estiver OFF ou se estiver freando/pausando, solta.
        if not self.accel_enabled or gesture in {"FREAR", "PAUSAR"}:
            self.release(accel_key)
            return

        self.press(accel_key)
