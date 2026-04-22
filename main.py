import cv2
import time
import threading
import os
import sys
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

import config
from src.hand_tracker import HandTracker
from src.gesture_detector import identify_gesture

# Tentar importar GamepadMapper, senão usar KeyboardMapper como fallback
try:
    import vgamepad as vg
    from src.gamepad_mapper import GamepadMapper as Mapper
except ImportError:
    print("AVISO: vgamepad não instalado. Usando KeyboardMapper como fallback.")
    from src.keyboard_mapper import KeyboardMapper as Mapper

# ───────── ESTADO GLOBAL PARA O DASHBOARD ─────────
_latest_jpeg = None
_frame_lock  = threading.Lock()
_shared_state = {
    "gesture": "NEUTRO",
    "fps": 0,
    "hands_detected": 0,
    "left_detected": False,
    "right_detected": False
}
_state_lock = threading.Lock()

# ───────── SERVIDOR DE EVENTOS (SSE) ─────────

class GestureEventHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global _latest_jpeg
        if self.path == '/events':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            try:
                while True:
                    with _state_lock:
                        data = json.dumps(_shared_state)
                    self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(1 / config.TARGET_FPS)
            except (BrokenPipeError, ConnectionResetError):
                pass

        elif self.path == '/video':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            try:
                while True:
                    with _frame_lock:
                        jpeg = _latest_jpeg
                    if jpeg is None:
                        time.sleep(0.01)
                        continue

                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                    self.wfile.write(jpeg)
                    self.wfile.write(b'\r\n')
                    self.wfile.flush()
                    time.sleep(1 / config.TARGET_FPS)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()

def start_sse_server():
    try:
        server = HTTPServer((config.SSE_HOST, config.SSE_PORT), GestureEventHandler)
        print(f"INFO: Servidor de Monitoramento em http://{config.SSE_HOST}:{config.SSE_PORT}")
        server.serve_forever()
    except Exception as e:
        print(f"AVISO: Servidor de monitoramento não iniciado: {e}")

# ───────── HUD E DESIGN ─────────

def draw_hud(frame, state: dict, fps: float, hands_detected: dict):
    h, w, _ = frame.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    gesture = state.get("action", "NEUTRO")
    if gesture == "NEUTRO":
        gesture = state.get("steering", "NEUTRO")
    
    color = config.GESTURE_COLOR_MAP.get(gesture, (180, 180, 180))
    
    # ── Caixa de Status Principal ──
    cv2.rectangle(frame, (10, 10), (320, 160), (30, 30, 30), -1)
    cv2.rectangle(frame, (10, 10), (320, 160), color, 2)
    
    cv2.putText(frame, f"CONTROLE: {gesture}", (20, 45), font, 0.8, color, 2)
    
    # Telemetria Detalhada
    accel_val = state.get("hand_height", 1.0)
    # Inverter e normalizar para exibição (0.3 topo, 0.75 base)
    motor_pct = max(0, min(100, int((0.75 - accel_val) / (0.75 - 0.3) * 100)))
    if config.CALIBRATION_MODE: motor_pct = 0
    
    cv2.putText(frame, f"MOTOR: {motor_pct}%", (20, 80), font, 0.6, (255, 255, 255), 1)
    
    angle = state.get("steering_angle", 0.0)
    cv2.putText(frame, f"ANGULO: {angle:+.1f} DEG", (20, 110), font, 0.6, (255, 255, 255), 1)
    
    pedal = state.get("pedal", "NEUTRO")
    cv2.putText(frame, f"PEDAL: {pedal}", (20, 140), font, 0.6, (0, 255, 255) if pedal != "NEUTRO" else (150, 150, 150), 1)

    # ── Info de Sistema (Direita) ──
    cv2.putText(frame, f"FPS: {int(fps)}", (w - 100, 30), font, 0.6, (0, 255, 0), 1)
    if config.CALIBRATION_MODE:
        cv2.putText(frame, "MODO CALIBRACAO", (w - 200, 60), font, 0.6, (0, 165, 255), 2)

    # ── Linha de Aceleração ──
    line_y = int(h * 0.75)
    cv2.line(frame, (0, line_y), (w, line_y), (100, 100, 100), 1)
    cv2.putText(frame, "LIMITE MOTOR", (w - 120, line_y - 10), font, 0.4, (100, 100, 100), 1)

def draw_reference_lines(frame):
    h, w, _ = frame.shape
    mid_x = w // 2
    cv2.line(frame, (mid_x, 0), (mid_x, h), (50, 50, 50), 1)

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    print("\n" + "="*50)
    print("   CONTROLADOR POR GESTOS - MARIO KART (AI)")
    print("="*50)
    print("INFO: Carregando módulos de visão...")
    
    tracker = HandTracker()
    mapper  = Mapper()
    print(f"INFO: Dispositivo de entrada: {type(mapper).__name__}")
    
    sse_thread = threading.Thread(target=start_sse_server, daemon=True)
    sse_thread.start()
    
    cv2.namedWindow("Mario Kart Controller")
    print("INFO: Sistema pronto! Pressione ESC para sair.\n")

    try:
        while True:
            # Esperar o próximo frame da thread de captura
            if not tracker.image_ready:
                time.sleep(0.001)
                continue
            
            frame = tracker.image_from_thread.copy()
            tracker.image_ready = False

            # Espelhar imagem se configurado
            if config.PREVIEW_MIRRORED:
                frame = cv2.flip(frame, 1)

            # Processamento
            result, _ = tracker.process_frame(frame)
            hands     = tracker.get_hands_by_label(result)
            
            # Lógica de Controle
            left  = hands.get("Left")
            right = hands.get("Right")
            gesture_state = identify_gesture(left, right)
            
            # Aplicar comandos
            mapper.update(gesture_state)

            # Desenhar Esqueleto das Mãos
            if config.DRAW_HAND_LANDMARKS:
                for hand_landmarks in hands.values():
                    if hand_landmarks:
                        tracker.draw_landmarks(frame, hand_landmarks)

            # Atualizar Estado Global (SSE)
            with _state_lock:
                _shared_state["gesture"] = gesture_state.get("action", gesture_state.get("steering", "NEUTRO"))
                _shared_state["fps"] = int(tracker.get_fps())
                _shared_state["hands_detected"] = len(hands)

            # MJPEG para o vídeo do Dashboard
            if _frame_lock.acquire(blocking=False):
                try:
                    _, buffer = cv2.imencode('.jpg', frame)
                    global _latest_jpeg
                    _latest_jpeg = buffer.tobytes()
                finally:
                    _frame_lock.release()

            # Desenhar Interface
            draw_hud(frame, gesture_state, tracker.get_fps(), hands)
            draw_reference_lines(frame)
            
            cv2.imshow("Mario Kart Controller", frame)
            if cv2.waitKey(1) == 27: 
                print("INFO: Encerrando pelo usuário (ESC)...")
                break

    except KeyboardInterrupt:
        print("\nINFO: Encerrando por interrupção do sistema...")
    finally:
        tracker.release()
        cv2.destroyAllWindows()
        print("INFO: Todos os recursos foram liberados com sucesso. Ate logo!")

if __name__ == "__main__":
    main()
