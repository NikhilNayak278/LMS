from enum import Enum
from pydantic import BaseModel

class CognitiveState(str, Enum):
    FOCUSED = "Focused"
    FRUSTRATED = "Frustrated"
    IDLE = "Idle"

class TelemetryData(BaseModel):
    scroll_velocity: float
    dwell_time: float
    tab_switches: int

def analyze_student_state(data: TelemetryData) -> CognitiveState:
    """
    Heuristic-based Mock Model for Cognitive Load detection.
    
    Frustrated: High scroll velocity (frantic), low dwell time, or high tab switches.
    Idle: Low scroll velocity, high dwell time, high tab switches.
    Focused: Moderate/low scroll velocity, high dwell time, low tab switches.
    """
    
    if data.tab_switches > 5 or data.scroll_velocity > 100:
        return CognitiveState.FRUSTRATED
    elif data.dwell_time > 120 and data.scroll_velocity < 5:
        return CognitiveState.IDLE
    else:
        return CognitiveState.FOCUSED
