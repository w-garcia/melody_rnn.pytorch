"""
Constants for processing midi.
Based on original constants.py in magenta
"""

# Standard pulses per quarter.
# https://en.wikipedia.org/wiki/Pulses_per_quarter_note
STANDARD_PPQ = 220

# Meter-related constants.
DEFAULT_QUARTERS_PER_MINUTE = 120.0
DEFAULT_STEPS_PER_BAR = 16  # 4/4 music sampled at 4 steps per quarter note.
DEFAULT_STEPS_PER_QUARTER = 4