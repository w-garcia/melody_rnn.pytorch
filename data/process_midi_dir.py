# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r""""Converts music files to NoteSequence protos and writes TFRecord file.

Currently supports MIDI (.mid, .midi) files.
"""

import os
import logging
import data.utils_midi as midi_io
import data.utils_sequences as note_sequence_io
import data.utils_melodies as melodies_io
import data.utils_encode_decode as coding_io

import fnmatch
import argparse

from data.constants import *

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def convert_directory(root_dir, out_dir):
    """Converts files in root_dir.

    Args:
      root_dir: A string specifying a root directory of midi files.
      out_dir: string specifying the output directory to store dataset object.
    Returns:
      A map from the resulting Futures to the file paths being converted.
    """
    logger.info("Converting files in '%s'.", root_dir)
    encoded_midis = []
    written_count = 0

    for root, dirs, filenames in os.walk(root_dir):
        for f in fnmatch.filter(filenames, '*.midi') + fnmatch.filter(filenames, '*.mid'):
            full_file_path = os.path.join(root, f)

            if written_count % 100 == 0:
                logger.debug("Processed {} files so far".format(written_count))

            try:
                sequence = convert_midi(root_dir, full_file_path)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("({}) generated an exception:".format(full_file_path))
                logger.exception(exc)
                continue

            if not sequence:
                continue

            # Getting closer
            # The NoteSequence proto becomes a SequenceExample now. Have to mimic directed acyclic graph

            # I guess this is the syntax:
            # dag[op] = input
            # dag[next_op] = op (<- out from previous line)
            # dag[output] = next_op
            # Becomes input -> op -> next_op -> output

            # Pipeline to recreate: melody_rnn_create_dataset.py:89
            # dag[time_change_splitter] = partitioner[mode + '_melodies']
            # dag[quantizer] = time_change_splitter
            # dag[melody_extractor] = quantizer
            # dag[encoder_pipeline] = melody_extractor
            # dag[dag_pipeline.DagOutput(mode + '_melodies')] = encoder_pipeline

            # If it is, then the order is:
            # NoteSequence -> partitioner -> splitter -> quantizer -> melody extractor -> encoder -> output
            # Don't need partitioner for now...

            # time_change_splitter: note_sequence_pipelines.TimeChangeSplitter
            split_seqs = note_sequence_io.split_note_sequence_on_time_changes(sequence)
            # take first one, maybe in most cases its just 1 time signature
            split_seqs = split_seqs[0]

            # quantizer: note_sequence_pipelines.Quantizer
            steps_per_quarter = 4  # just felt like putting 4
            quantized_sequence = note_sequence_io.quantize_note_sequence(split_seqs, steps_per_quarter)
            if not quantized_sequence:
                continue
            # TODO: Handle a bunch of exceptions here

            # Here is where it becomes a Melody object
            # melody_extractor: melody_pipelines.MelodyExtractor
            # music.melodies_lib.py:675 implies I am on the right path
            melodies = melodies_io.extract_melodies(quantized_sequence)
            if melodies == []:
                continue
            #TODO: Handle more exceptions here (melody_pipeline.py)

            # Now to SequenceExample, finally
            # encoder: EncoderPipeline
            # dag is not clear how to handle list of melodies. Just take first instrument for no
            melody = melodies[0]
            melody.squash(DEFAULT_MIN_NOTE, DEFAULT_MAX_NOTE, DEFAULT_TRANSPOSE_TO_KEY)
            melody_encoding = coding_io.MelodyOneHotEncoding(DEFAULT_MIN_NOTE, DEFAULT_MAX_NOTE)
            transcoder = coding_io.OneHotEventSequenceEncoderDecoder(melody_encoding)
            encoded = transcoder.encode(melody)
            if not encoded:
                continue

            encoded_midis.append(encoded)
            #TODO: Need to create dataset file.
            written_count += 1

    return encoded_midis


def convert_midi(root_dir, full_file_path):
    """Converts a midi file to a sequence proto.

    Args:
      root_dir: A string specifying the root directory for the files being
          converted.
      sub_dir: The directory being converted currently.
      full_file_path: the full path to the file to convert.

    Returns:
      Either a NoteSequence proto or None if the file could not be converted.
    """
    try:
        sequence = midi_io.midi_file_to_sequence_proto(full_file_path)
    except midi_io.MIDIConversionError as e:
        logger.warning('Could not parse MIDI file %s. It will be skipped. Error was: %s', full_file_path, e)
        return None
    sequence.collection_name = os.path.basename(root_dir)
    sequence.filename = full_file_path  # maybe this can be the full path instead ?? os.path.join(sub_dir, os.path.basename(full_file_path))
    sequence.id = note_sequence_io.generate_note_sequence_id(sequence.filename, sequence.collection_name, 'midi')
    logger.info('Converted MIDI file %s.', full_file_path)
    return sequence


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', help="Directory of midi files to process.", required=True)
    ap.add_argument('--out_dir', help="Path to out dir.", required=True)
    ap.add_argument('--out_path', help="Path to finished pkl file.", required=True)

    args = ap.parse_args()

    convert_directory(args.input_dir, args.out_dir)
