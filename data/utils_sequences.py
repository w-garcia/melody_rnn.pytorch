"""
Consolidation of utils fns from magenta.
"""
import hashlib
import data.protobuf.music_pb2 as music_pb2
import data.constants as constants
from data.exceptions import *


# Set the quantization cutoff.
# Note events before this cutoff are rounded down to nearest step. Notes
# above this cutoff are rounded up to nearest step. The cutoff is given as a
# fraction of a step.
# For example, with quantize_cutoff = 0.75 using 0-based indexing,
# if .75 < event <= 1.75, it will be quantized to step 1.
# If 1.75 < event <= 2.75 it will be quantized to step 2.
# A number close to 1.0 gives less wiggle room for notes that start early,
# and they will be snapped to the previous step.
QUANTIZE_CUTOFF = 0.5

# Shortcut to chord symbol text annotation type.
CHORD_SYMBOL = music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL


def generate_note_sequence_id(filename, collection_name, source_type):
    """Generates a unique ID for a sequence.

    The format is:'/id/<type>/<collection name>/<hash>'.

    Args:
      filename: The string path to the source file relative to the root of the
          collection.
      collection_name: The collection from which the file comes.
      source_type: The source type as a string (e.g. "midi" or "abc").

    Returns:
      The generated sequence ID as a string.
    """
    # TODO(adarob): Replace with FarmHash when it becomes a part of TensorFlow.
    filename_fingerprint = hashlib.sha1(filename.encode('utf-8'))
    return '/id/%s/%s/%s' % (
        source_type.lower(), collection_name, filename_fingerprint.hexdigest())


def is_quantized_sequence(note_sequence):
    """Returns whether or not a NoteSequence proto has been quantized.

    Args:
      note_sequence: A music_pb2.NoteSequence proto.

    Returns:
      True if `note_sequence` is quantized, otherwise False.
    """
    # If the QuantizationInfo message has a non-zero steps_per_quarter or
    # steps_per_second, assume that the proto has been quantized.
    return (note_sequence.quantization_info.steps_per_quarter > 0 or
            note_sequence.quantization_info.steps_per_second > 0)


def extract_subsequence(sequence, start_time, end_time, sustain_control_number=64):
    """Extracts a subsequence from a NoteSequence.

    Notes starting before `start_time` are not included. Notes ending after
    `end_time` are truncated. Time signature, tempo, key signature, chord changes,
    and sustain pedal events outside the specified time range are removed;
    however, the most recent event of each of these types prior to `start_time` is
    included at `start_time`. This means that e.g. if a time signature of 3/4 is
    specified in the original sequence prior to `start_time` (and is not followed
    by a different time signature), the extracted subsequence will include a 3/4
    time signature event at `start_time`. Pitch bends and control changes other
    than sustain are removed entirely.

    The extracted subsequence is shifted to start at time zero.

    Args:
      sequence: The NoteSequence to extract a subsequence from.
      start_time: The float time in seconds to start the subsequence.
      end_time: The float time in seconds to end the subsequence.
      sustain_control_number: The MIDI control number for sustain pedal.

    Returns:
      A new NoteSequence containing the subsequence of `sequence` from the
      specified time range.

    Raises:
      QuantizationStatusException: If the sequence has already been quantized.
      ValueError: If `start_time` is past the end of `sequence`.
    """
    if is_quantized_sequence(sequence):
        raise QuantizationStatusException(
            'Can only extract subsequence from unquantized NoteSequence.')

    if start_time >= sequence.total_time:
        raise ValueError('Cannot extract subsequence past end of sequence.')

    subsequence = music_pb2.NoteSequence()
    subsequence.CopyFrom(sequence)

    subsequence.total_time = 0.0

    # Extract notes.
    del subsequence.notes[:]
    for note in sequence.notes:
        if note.start_time < start_time or note.start_time >= end_time:
            continue
        new_note = subsequence.notes.add()
        new_note.CopyFrom(note)
        new_note.start_time -= start_time
        new_note.end_time = min(note.end_time, end_time) - start_time
        if new_note.end_time > subsequence.total_time:
            subsequence.total_time = new_note.end_time

    # Extract time signatures, key signatures, tempos, and chord changes (other
    # text annotations are deleted).

    del subsequence.time_signatures[:]
    del subsequence.key_signatures[:]
    del subsequence.tempos[:]
    del subsequence.text_annotations[:]

    event_types = [
        music_pb2.NoteSequence.TimeSignature, music_pb2.NoteSequence.KeySignature,
        music_pb2.NoteSequence.Tempo, music_pb2.NoteSequence.TextAnnotation]
    events_by_type = [
        sequence.time_signatures, sequence.key_signatures, sequence.tempos,
        [annotation for annotation in sequence.text_annotations
         if annotation.annotation_type == CHORD_SYMBOL]]
    new_event_containers = [
        subsequence.time_signatures, subsequence.key_signatures,
        subsequence.tempos, subsequence.text_annotations]

    for event_type, events, container in zip(
            event_types, events_by_type, new_event_containers):
        initial_event = None
        for event in sorted(events, key=lambda event: event.time):
            if event.time <= start_time:
                initial_event = event_type()
                initial_event.CopyFrom(event)
                continue
            elif event.time >= end_time:
                break
            new_event = container.add()
            new_event.CopyFrom(event)
            new_event.time -= start_time
        if initial_event:
            initial_event.time = 0.0
            container.extend([initial_event])
        container.sort(key=lambda event: event.time)

    # Extract sustain pedal events (other control changes are deleted). Sustain
    # pedal state prior to the extracted subsequence is maintained per-instrument.
    del subsequence.control_changes[:]
    sustain_events = [cc for cc in sequence.control_changes
                      if cc.control_number == sustain_control_number]
    initial_sustain_events = {}
    for sustain_event in sorted(sustain_events, key=lambda event: event.time):
        if sustain_event.time <= start_time:
            initial_sustain_event = music_pb2.NoteSequence.ControlChange()
            initial_sustain_event.CopyFrom(sustain_event)
            initial_sustain_events[sustain_event.instrument] = initial_sustain_event
            continue
        elif sustain_event.time >= end_time:
            break
        new_sustain_event = subsequence.control_changes.add()
        new_sustain_event.CopyFrom(sustain_event)
        new_sustain_event.time -= start_time
    for _, initial_sustain_event in initial_sustain_events.items():
        initial_sustain_event.time = 0.0
        subsequence.control_changes.extend([initial_sustain_event])
    subsequence.control_changes.sort(key=lambda cc: cc.time)

    # Pitch bends are deleted entirely.
    del subsequence.pitch_bends[:]

    subsequence.subsequence_info.start_time_offset = start_time
    subsequence.subsequence_info.end_time_offset = (
            sequence.total_time - start_time - subsequence.total_time)

    return subsequence


def split_note_sequence_on_time_changes(note_sequence, skip_splits_inside_notes=False):
    """Split one NoteSequence into many around time signature and tempo changes.

    This function splits a NoteSequence into multiple NoteSequences, each of which
    contains only a single time signature and tempo, unless `split_notes` is False
    in which case all time signature and tempo changes occur within sustained
    notes. Each of the resulting NoteSequences is shifted to start at time zero.

    Args:
      note_sequence: The NoteSequence to split.
      skip_splits_inside_notes: If False, the NoteSequence will be split at all
          time changes, regardless of whether or not any notes are sustained
          across the time change. If True, the NoteSequence will not be split at
          time changes that occur within sustained notes.

    Returns:
      A Python list of NoteSequences.
    """
    prev_change_time = 0.0

    current_numerator = 4
    current_denominator = 4
    current_qpm = constants.DEFAULT_QUARTERS_PER_MINUTE

    time_signatures_and_tempos = sorted(
        list(note_sequence.time_signatures) + list(note_sequence.tempos),
        key=lambda t: t.time)
    time_signatures_and_tempos = [t for t in time_signatures_and_tempos
                                  if t.time < note_sequence.total_time]

    notes_by_start_time = sorted(list(note_sequence.notes),
                                 key=lambda note: note.start_time)
    note_idx = 0
    notes_crossing_split = []

    subsequences = []

    for time_change in time_signatures_and_tempos:
        if isinstance(time_change, music_pb2.NoteSequence.TimeSignature):
            if (time_change.numerator == current_numerator and
                    time_change.denominator == current_denominator):
                # Time signature didn't actually change.
                continue
        else:
            if time_change.qpm == current_qpm:
                # Tempo didn't actually change.
                continue

        # Update notes crossing potential split.
        while (note_idx < len(notes_by_start_time) and
               notes_by_start_time[note_idx].start_time < time_change.time):
            notes_crossing_split.append(notes_by_start_time[note_idx])
            note_idx += 1
        notes_crossing_split = [note for note in notes_crossing_split
                                if note.end_time > time_change.time]

        if time_change.time > prev_change_time:
            if not (skip_splits_inside_notes and notes_crossing_split):
                # Extract the subsequence between the previous time change and this
                # time change.
                subsequence = extract_subsequence(note_sequence, prev_change_time,
                                                  time_change.time)
                subsequences.append(subsequence)
                prev_change_time = time_change.time

        # Even if we didn't split here, update the current time signature or tempo.
        if isinstance(time_change, music_pb2.NoteSequence.TimeSignature):
            current_numerator = time_change.numerator
            current_denominator = time_change.denominator
        else:
            current_qpm = time_change.qpm

    # Handle the final subsequence.
    if note_sequence.total_time > prev_change_time:
        subsequence = extract_subsequence(note_sequence, prev_change_time,
                                          note_sequence.total_time)
        subsequences.append(subsequence)

    return subsequences
