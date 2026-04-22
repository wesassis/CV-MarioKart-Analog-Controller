import vgamepad as vg
import config

class GamepadMapper:
    """
    Mapeia gestos analógicos para um controle virtual de Xbox 360.
    Inspirado na lógica de eixos do PRD seção 6.2.
    """
    def __init__(self):
        print("INFO: Inicializando Gamepad Virtual (Xbox 360)...")
        try:
            self.gamepad = vg.VX360Gamepad()
            print("INFO: Gamepad Virtual criado com sucesso!")
        except Exception as e:
            print(f"ERRO: Não foi possível criar o gamepad: {e}")
            raise e

    def update(self, state: dict):
        """
        Recebe o estado analógico e atualiza os eixos do gamepad.
        """
        # ─── 1. VOLANTE ANALÓGICO (Eixo X) ───
        raw_angle = state.get("steering_angle", 0.0)
        limit = 30.0
        normalized_steering = max(-1.0, min(1.0, raw_angle / limit))
        self.gamepad.left_joystick_float(x_value_float=normalized_steering, y_value_float=0.0)

        # ─── 2. ACELERAÇÃO ANALÓGICA (Gatilho RT) ───
        height = state.get("hand_height", 1.0)
        accel_val = 0.0
        if not getattr(config, "CALIBRATION_MODE", False):
            if height < 0.75:
                # Interpolação: 0.3 (topo) -> 100%, 0.75 (limite) -> 0%
                accel_val = max(0.0, min(1.0, (0.75 - height) / (0.75 - 0.30)))
        
        self.gamepad.right_trigger_float(value_float=accel_val)

        # ─── 3. AÇÕES DIGITAIS (BOTOES XBOX) ───
        action = state.get("action", "NEUTRO")
        pedal = state.get("pedal", "NEUTRO")

        # Botão A (Xbox) -> Digital para Aceleração
        if accel_val > 0.5:
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        else:
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)

        # Botão B (Xbox) -> Freio
        if pedal == "FREAR":
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        else:
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)

        # Botão X (Xbox) -> Item (Mão Esquerda V) -> Mapear no 'Y' do Dolphin
        if action == "ITEM":
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        else:
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)

        # Botão Y (Xbox) -> Trocar (Mão Direita V) -> Mapear no 'Z' do Dolphin
        if action == "TROCAR":
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
        else:
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)

        # Botão START (Xbox) -> Pausar
        if action == "PAUSAR":
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
        else:
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)

        self.gamepad.update()

        # ─── DEBUG DETALHADO EM PT-BR (A cada 30 frames) ───
        if not hasattr(self, "_frame_count"): self._frame_count = 0
        self._frame_count += 1
        if self._frame_count % 30 == 0:
            print(f"DEBUG CONTROLE | Volante: {normalized_steering:+.2f} ({raw_angle:+.1f}°) | Motor: {int(accel_val*100)}% | Pedal: {pedal} | Ação: {action}")
