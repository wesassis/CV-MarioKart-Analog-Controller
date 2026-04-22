"""
hand_tracker.py — Wrapper MediaPipe Hands com câmera em thread separada
Inspirado em: CameraStream class do helpers.py (VR Tracking) + inicia_mediapipe() do cognitive ref
Padrão: image_ready flag + daemon thread + cleanup explícito
"""

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlretrieve

import cv2
import mediapipe as mp

import config

HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


class HandTracker:
    """
    Captura frames da webcam em thread separada e processa com MediaPipe Hands.
    Mantém o padrão image_ready flag do CameraStream do VR Tracking.
    """

    def __init__(self):
        self._use_tasks = not hasattr(mp, "solutions")

        # ── MediaPipe (Solutions API, legado) ──
        if not self._use_tasks:
            mp_hands = mp.solutions.hands
            self.hands = mp_hands.Hands(
                max_num_hands=config.MAX_NUM_HANDS,
                min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_hands_ref = mp_hands
            self._task_connections = None
        else:
            # ── MediaPipe Tasks API (Python 3.13+) ──
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_path = self._ensure_hand_model()
            options = mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=config.MAX_NUM_HANDS,
                min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                min_hand_presence_confidence=config.MIN_TRACKING_CONFIDENCE,
                min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            )
            self.hands = mp_vision.HandLandmarker.create_from_options(options)
            self._task_connections = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS
            self.mp_drawing = None
            self.mp_hands_ref = None

        # ── Thread de câmera (igual CameraStream) ──
        self.image_ready        = False
        self.image_from_thread  = None
        self._running           = True
        self._exit_ready        = False

        self.cap = self._open_camera()

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          config.TARGET_FPS)

        print("INFO: Iniciando thread de câmera...")
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    # ───────── THREAD DE CAPTURA ─────────

    def _update(self):
        """Loop de captura contínua — igual ao CameraStream.update() do VR Tracking"""
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                print("ERROR: Falha na captura de frame.")
                self._exit_ready = True
                return
            self.image_from_thread = frame
            self.image_ready = True

    # ───────── PROCESSAMENTO DO FRAME ─────────

    def process_frame(self, frame):
        """
        Processa um frame BGR com MediaPipe Hands.
        Retorna (result, frame_rgb) — igual ao padrão dos refs.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        if self._use_tasks:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            task_result = self.hands.detect(mp_image)
            result = self._convert_task_result(task_result)
        else:
            result = self.hands.process(frame_rgb)

        frame_rgb.flags.writeable = True
        return result, frame_rgb

    def draw_landmarks(self, frame, hand_landmarks):
        """Desenha landmarks e conexões no frame (igual mp_drawing.draw_landmarks dos refs)"""
        if not self._use_tasks:
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands_ref.HAND_CONNECTIONS,
            )
            return

        h, w, _ = frame.shape
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

        for conn in self._task_connections:
            p1 = points[conn.start]
            p2 = points[conn.end]
            cv2.line(frame, p1, p2, (0, 180, 255), 2)

        for x, y in points:
            cv2.circle(frame, (x, y), 3, (232, 54, 74), -1)

    # ───────── IDENTIFICAÇÃO DE MÃO (L/R) ─────────

    def get_hands_by_label(self, result):
        """
        Retorna dicionário {'Left': landmarks, 'Right': landmarks}.
        Prioriza posição X no frame espelhado para consistência de UX.
        """
        hands = {"Left": None, "Right": None}

        if not result.multi_hand_landmarks:
            return hands

        detected = []
        for hand_landmarks, handedness in zip(
            result.multi_hand_landmarks,
            result.multi_handedness,
        ):
            label = handedness.classification[0].label   # fallback ('Left' ou 'Right')
            center_x = sum(lm.x for lm in hand_landmarks.landmark) / len(hand_landmarks.landmark)
            detected.append((label, hand_landmarks, center_x))

        # 1 mão: usa lado visual para decidir Left/Right do usuário
        if len(detected) == 1:
            _, hand_landmarks, center_x = detected[0]
            if config.PREVIEW_MIRRORED:
                mapped = "Left" if center_x >= 0.5 else "Right"
            else:
                mapped = "Right" if center_x >= 0.5 else "Left"
            hands[mapped] = hand_landmarks
            return hands

        # 2+ mãos: ordena por X para evitar troca de label frame a frame
        sorted_by_x = sorted(detected, key=lambda item: item[2])  # menor X -> lado esquerdo da imagem
        if config.PREVIEW_MIRRORED:
            # imagem espelhada: menor X = mão direita do usuário; maior X = mão esquerda
            hands["Right"] = sorted_by_x[0][1]
            hands["Left"] = sorted_by_x[-1][1]
        else:
            hands["Left"] = sorted_by_x[0][1]
            hands["Right"] = sorted_by_x[-1][1]

        return hands

    # ───────── PROPRIEDADES E PERFORMANCE ─────────
    
    def get_fps(self) -> float:
        """Calcula e retorna o FPS atual do processamento."""
        if not hasattr(self, "_last_time"):
            self._last_time = time.time()
            return 0.0
        
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        return 1.0 / dt if dt > 0 else 0.0

    @property
    def should_exit(self) -> bool:
        return self._exit_ready

    # ───────── CLEANUP ─────────

    def release(self):
        """Libera recursos da câmera e MediaPipe"""
        self._running = False
        self.hands.close()
        self.cap.release()
        print("INFO: HandTracker liberado.")

    # ───────── TASKS HELPERS ─────────

    def _ensure_hand_model(self) -> Path:
        """Garante presença local do modelo hand_landmarker.task para Tasks API."""
        model_dir = Path(__file__).resolve().parents[1] / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "hand_landmarker.task"

        if not model_path.exists():
            print("INFO: Baixando modelo hand_landmarker.task...")
            urlretrieve(HAND_LANDMARKER_MODEL_URL, str(model_path))

        return model_path

    def _open_camera(self):
        """Tenta abrir webcam por índices comuns (0, 1, 2)."""
        for cam_index in (0, 1, 2):
            cap = cv2.VideoCapture(cam_index)
            if cap.isOpened():
                print(f"INFO: Webcam aberta no índice {cam_index}.")
                return cap
            cap.release()
        raise RuntimeError("ERROR: Não foi possível abrir a câmera (índices 0, 1, 2).")

    def _convert_task_result(self, task_result):
        """Converte resultado da Tasks API para shape compatível com Solutions API."""
        multi_hand_landmarks = []
        multi_handedness = []

        for idx, landmarks in enumerate(task_result.hand_landmarks):
            multi_hand_landmarks.append(SimpleNamespace(landmark=landmarks))

            label = "Right"
            if idx < len(task_result.handedness) and task_result.handedness[idx]:
                category = task_result.handedness[idx][0]
                label = category.category_name or label

            multi_handedness.append(
                SimpleNamespace(classification=[SimpleNamespace(label=label)])
            )

        return SimpleNamespace(
            multi_hand_landmarks=multi_hand_landmarks if multi_hand_landmarks else None,
            multi_handedness=multi_handedness if multi_handedness else None,
        )
