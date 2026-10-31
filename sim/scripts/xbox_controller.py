import time
from dataclasses import dataclass

import _paths  # noqa: F401  (control/ no sys.path)
from repo_control_policy import NOMINAL_LEG_LENGTH, RobotCommand


def deadzone(value, threshold):
    return 0.0 if abs(value) < threshold else value


def axis_value(joystick, index, default=0.0):
    if index is None or index < 0 or index >= joystick.get_numaxes():
        return default
    return float(joystick.get_axis(index))


def button_value(joystick, index):
    if index is None or index < 0 or index >= joystick.get_numbuttons():
        return False
    return bool(joystick.get_button(index))


def hat_y(joystick, index=0):
    """D-pad vertical: +1 up, -1 down, 0 centered. Robust across controllers."""
    try:
        if index < joystick.get_numhats():
            return float(joystick.get_hat(index)[1])
    except Exception:
        pass
    return 0.0


@dataclass
class XboxMapping:
    speed_axis: int = 1       # left stick Y (vertical) -> forward/back
    yaw_axis: int = 2         # right stick X (horizontal) -> yaw
    left_trigger_axis: int = 4
    right_trigger_axis: int = 5
    hat_index: int = 0        # D-pad up/down -> ride height (reliable fallback)
    stop_button: int = 0      # A: soft stop (zero the command, coast down)
    brake_button: int = 1     # B: hard brake (snap speed target to 0, max decel)
    reset_button: int = 7
    gear_button: int = 2      # X: alterna o trem de pouso (abaixado / recolhido)
    invert_speed: bool = True
    invert_yaw: bool = True    # right-stick-right turns right


class XboxController:
    def __init__(
        self,
        pygame,
        mapping=None,
        deadzone_value=0.12,
        max_speed=2.5,
        max_yaw_rate=1.8,
        max_leg_adjust=0.09,
        height_rate=0.12,
    ):
        self.pygame = pygame
        self.mapping = mapping or XboxMapping()
        self.deadzone = float(deadzone_value)
        self.max_speed = float(max_speed)
        self.max_yaw_rate = float(max_yaw_rate)
        self.max_leg_adjust = float(max_leg_adjust)
        # Triggers adjust a PERSISTENT leg-height offset: hold a trigger to ramp
        # the ride height and release to hold it there (it latches), rather than
        # only changing height while held. height_rate is m/s of leg length at
        # full trigger.
        self.height_rate = float(height_rate)
        self.height_offset = 0.0
        self._last_time = None
        self._reset_requested = False
        self.gear_down = True
        self._gear_prev = False

        pygame.init()
        count = pygame.joystick.get_count()
        if count < 1:
            raise RuntimeError("No gamepad detected by pygame. Pair the Xbox controller over Bluetooth and try again.")
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

    @property
    def name(self):
        return self.joystick.get_name()

    def read(self):
        self.pygame.event.pump()
        mapping = self.mapping

        speed_axis = axis_value(self.joystick, mapping.speed_axis)
        yaw_axis = axis_value(self.joystick, mapping.yaw_axis)
        if mapping.invert_speed:
            speed_axis = -speed_axis
        if mapping.invert_yaw:
            yaw_axis = -yaw_axis

        speed_axis = deadzone(speed_axis, self.deadzone)
        yaw_axis = deadzone(yaw_axis, self.deadzone)

        # Integrate height input into a persistent (latching) ride-height offset.
        # Two inputs are summed so it works regardless of trigger axis mapping:
        # the triggers (RT up / LT down) and the D-pad (up/down, very reliable).
        now = time.time()
        dt = 0.0 if self._last_time is None else max(0.0, min(0.1, now - self._last_time))
        self._last_time = now
        trig = self._trigger_01(mapping.right_trigger_axis) - self._trigger_01(mapping.left_trigger_axis)
        dpad = hat_y(self.joystick, mapping.hat_index)
        height_input = max(-1.0, min(1.0, trig + dpad))
        self.height_offset += height_input * self.height_rate * dt
        self.height_offset = max(-self.max_leg_adjust, min(self.max_leg_adjust, self.height_offset))

        if button_value(self.joystick, mapping.stop_button):
            speed_axis = 0.0
            yaw_axis = 0.0

        # Hard brake (B): zero the speed command and flag command.brake so the
        # controller snaps the speed target to 0 and applies maximum decel torque
        # (vs A, which just zeros the command and lets it coast down).
        braking = button_value(self.joystick, mapping.brake_button)
        if braking:
            speed_axis = 0.0

        gear_pressed = button_value(self.joystick, mapping.gear_button)
        if gear_pressed and not self._gear_prev:
            self.gear_down = not self.gear_down
        self._gear_prev = gear_pressed

        self._reset_requested = button_value(self.joystick, mapping.reset_button)
        if self._reset_requested:
            self.height_offset = 0.0
        return RobotCommand(
            speed_cmd=speed_axis * self.max_speed,
            yaw_speed_cmd=yaw_axis * self.max_yaw_rate,
            leg_length=NOMINAL_LEG_LENGTH + self.height_offset,
            brake=braking,
            gear_down=self.gear_down,
        )

    def consume_reset_requested(self):
        requested = self._reset_requested
        self._reset_requested = False
        return requested

    def _trigger_01(self, axis_index):
        value = axis_value(self.joystick, axis_index, default=-1.0)
        # Most SDL/Xbox trigger axes report -1 released and +1 pressed.
        return max(0.0, min(1.0, (value + 1.0) * 0.5))
