from xarm.wrapper import XArmAPI
from utils.constants import DEFAULT_SPEED, DEFAULT_ACCEL

class Lite6Controller:
    def __init__(self, ip_address):
        self.ip = ip_address
        self.arm = None

    def connect(self):
        """Connects to the arm, enables motion, and sets to position control."""
        print(f"[Robot] Connecting to Lite 6 at {self.ip}...")
        try:
            self.arm = XArmAPI(self.ip)
            self.arm.motion_enable(enable=True)
            self.arm.set_mode(0)  # Standard position control
            self.arm.set_state(state=0)
            print("[Robot] Successfully connected and enabled.")
        except Exception as e:
            print(f"[Robot] Connection failed: {e}")
            self.arm = None

    def disconnect(self):
        """Safely disconnects the arm."""
        if self.arm:
            self.arm.disconnect()
            print("[Robot] Disconnected.")

    def get_current_position(self):
        """Returns [x, y, z, roll, pitch, yaw] or None if failed."""
        if not self.arm: return None
        code, pos = self.arm.get_position()
        if code == 0:
            return pos
        return None

    def move_to(self, x, y, z, roll=-180.0, pitch=0.0, yaw=0.0, speed=DEFAULT_SPEED, wait=True):
        """Moves to a Cartesian coordinate safely."""
        if not self.arm:
            print("[Robot] Cannot move, not connected.")
            return

        print(f"[Robot] Moving to X:{x:.1f} Y:{y:.1f} Z:{z:.1f}...")
        self.arm.set_position(
            x=x, y=y, z=z, 
            roll=roll, pitch=pitch, yaw=yaw,
            speed=speed, mvacc=DEFAULT_ACCEL, wait=wait
        )