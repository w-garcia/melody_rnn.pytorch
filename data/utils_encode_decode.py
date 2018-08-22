import numpy as np
import abc
import torch

from data.constants import *


def make_sequence_example(inputs, labels):
    """Returns a SequenceExample for the given inputs and labels.

    Args:
      inputs: A list of input vectors. Each input vector is a list of floats.
      labels: A list of ints.

    Returns:
      A Pytorch tensor containing inputs and labels.
    """
    # input_features = [tf.train.Feature(float_list=tf.train.FloatList(value=input_)) for input_ in inputs]
    input_features = torch.Tensor(inputs)
    # label_features = [tf.train.Feature(int64_list=tf.train.Int64List(value=[label])) for label in labels]
    label_features = torch.Tensor(labels)

    # feature_list = {
    #     'inputs': tf.train.FeatureList(feature=input_features),
    #     'labels': tf.train.FeatureList(feature=label_features)
    # }
    # feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    return input_features, label_features


class OneHotEncoding(object):
  """An interface for specifying a one-hot encoding of individual events."""
  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def num_classes(self):
    """The number of distinct event encodings.

    Returns:
      An int, the range of ints that can be returned by self.encode_event.
    """
    pass

  @abc.abstractproperty
  def default_event(self):
    """An event value to use as a default.

    Returns:
      The default event value.
    """
    pass

  @abc.abstractmethod
  def encode_event(self, event):
    """Convert from an event value to an encoding integer.

    Args:
      event: An event value to encode.

    Returns:
      An integer representing the encoded event, in range [0, self.num_classes).
    """
    pass

  @abc.abstractmethod
  def decode_event(self, index):
    """Convert from an encoding integer to an event value.

    Args:
      index: The encoding, an integer in the range [0, self.num_classes).

    Returns:
      The decoded event value.
    """
    pass

  def event_to_num_steps(self, unused_event):
    """Returns the number of time steps corresponding to an event value.

    This is used for normalization when computing metrics. Subclasses with
    variable step size should override this method.

    Args:
      unused_event: An event value for which to return the number of steps.

    Returns:
      The number of steps corresponding to the given event value, defaulting to
      one.
    """
    return 1


class MelodyOneHotEncoding(OneHotEncoding):
    """Basic one hot encoding for melody events.

    Encodes melody events as follows:
      0 = no event,
      1 = note-off event,
      [2, self.num_classes) = note-on event for that pitch relative to the
          [self._min_note, self._max_note) range.
    """

    def __init__(self, min_note, max_note):
        """Initializes a MelodyOneHotEncoding object.

        Args:
          min_note: The minimum midi pitch the encoded melody events can have.
          max_note: The maximum midi pitch (exclusive) the encoded melody events
              can have.

        Raises:
          ValueError: If `min_note` or `max_note` are outside the midi range, or if
              `max_note` is not greater than `min_note`.
        """
        if min_note < MIN_MIDI_PITCH:
            raise ValueError('min_note must be >= 0. min_note is %d.' % min_note)
        if max_note > MAX_MIDI_PITCH + 1:
            raise ValueError('max_note must be <= 128. max_note is %d.' % max_note)
        if max_note <= min_note:
            raise ValueError('max_note must be greater than min_note')

        self._min_note = min_note
        self._max_note = max_note

    @property
    def num_classes(self):
        return self._max_note - self._min_note + NUM_SPECIAL_MELODY_EVENTS

    @property
    def default_event(self):
        return MELODY_NO_EVENT

    def encode_event(self, event):
        """Collapses a melody event value into a zero-based index range.

        Args:
          event: A Melody event value. -2 = no event, -1 = note-off event,
              [0, 127] = note-on event for that midi pitch.

        Returns:
          An int in the range [0, self.num_classes). 0 = no event,
          1 = note-off event, [2, self.num_classes) = note-on event for
          that pitch relative to the [self._min_note, self._max_note) range.

        Raises:
          ValueError: If `event` is a MIDI note not between self._min_note and
              self._max_note, or an invalid special event value.
        """
        if event < -NUM_SPECIAL_MELODY_EVENTS:
            raise ValueError('invalid melody event value: %d' % event)
        if (event >= 0) and (event < self._min_note):
            raise ValueError('melody event less than min note: %d < %d' % (
                event, self._min_note))
        if event >= self._max_note:
            raise ValueError('melody event greater than max note: %d >= %d' % (
                event, self._max_note))

        if event < 0:
            return event + NUM_SPECIAL_MELODY_EVENTS
        return event - self._min_note + NUM_SPECIAL_MELODY_EVENTS

    def decode_event(self, index):
        """Expands a zero-based index value to its equivalent melody event value.

        Args:
          index: An int in the range [0, self._num_model_events).
              0 = no event, 1 = note-off event,
              [2, self._num_model_events) = note-on event for that pitch relative
              to the [self._min_note, self._max_note) range.

        Returns:
          A Melody event value. -2 = no event, -1 = note-off event,
          [0, 127] = note-on event for that midi pitch.
        """
        if index < NUM_SPECIAL_MELODY_EVENTS:
            return index - NUM_SPECIAL_MELODY_EVENTS
        return index - NUM_SPECIAL_MELODY_EVENTS + self._min_note


class EventSequenceEncoderDecoder(object):
    """An abstract class for translating between events and model data.

    When building your dataset, the `encode` method takes in an event sequence
    and returns a SequenceExample of inputs and labels. These SequenceExamples
    are fed into the model during training and evaluation.

    During generation, the `get_inputs_batch` method takes in a list of the
    current event sequences and returns an inputs batch which is fed into the
    model to predict what the next event should be for each sequence. The
    `extend_event_sequences` method takes in the list of event sequences and the
    softmax returned by the model and extends each sequence by one step by
    sampling from the softmax probabilities. This loop (`get_inputs_batch` ->
    inputs batch is fed through the model to get a softmax ->
    `extend_event_sequences`) is repeated until the generated event sequences
    have reached the desired length.

    Properties:
      input_size: The length of the list returned by self.events_to_input.
      num_classes: The range of ints that can be returned by
          self.events_to_label.

    The `input_size`, `num_classes`, `events_to_input`, `events_to_label`, and
    `class_index_to_event` method must be overwritten to be specific to your
    model.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def input_size(self):
        """The size of the input vector used by this model.

        Returns:
            An integer, the length of the list returned by self.events_to_input.
        """
        pass

    @abc.abstractproperty
    def num_classes(self):
        """The range of labels used by this model.

        Returns:
            An integer, the range of integers that can be returned by
                self.events_to_label.
        """
        pass

    @abc.abstractproperty
    def default_event_label(self):
        """The class label that represents a default event.

        Returns:
          An int, the class label that represents a default event.
        """
        pass

    @abc.abstractmethod
    def events_to_input(self, events, position):
        """Returns the input vector for the event at the given position.

        Args:
          events: A list-like sequence of events.
          position: An integer event position in the sequence.

        Returns:
          An input vector, a self.input_size length list of floats.
        """
        pass

    @abc.abstractmethod
    def events_to_label(self, events, position):
        """Returns the label for the event at the given position.

        Args:
          events: A list-like sequence of events.
          position: An integer event position in the sequence.

        Returns:
          A label, an integer in the range [0, self.num_classes).
        """
        pass

    @abc.abstractmethod
    def class_index_to_event(self, class_index, events):
        """Returns the event for the given class index.

        This is the reverse process of the self.events_to_label method.

        Args:
          class_index: An integer in the range [0, self.num_classes).
          events: A list-like sequence of events.

        Returns:
          An event value.
        """
        pass

    def labels_to_num_steps(self, labels):
        """Returns the total number of time steps for a sequence of class labels.

        This is used for normalization when computing metrics. Subclasses with
        variable step size should override this method.

        Args:
          labels: A list-like sequence of integers in the range
              [0, self.num_classes).

        Returns:
          The total number of time steps for the label sequence, defaulting to one
          per event.
        """
        return len(labels)

    def encode(self, events):
        """Returns a SequenceExample for the given event sequence.

        Args:
          events: A list-like sequence of events.

        Returns:
          A Pytorch tensor containing inputs and labels.
        """
        inputs = []
        labels = []
        for i in range(len(events) - 1):
            inputs.append(self.events_to_input(events, i))
            labels.append(self.events_to_label(events, i + 1))
        return make_sequence_example(inputs, labels)

    def get_inputs_batch(self, event_sequences, full_length=False):
        """Returns an inputs batch for the given event sequences.

        Args:
          event_sequences: A list of list-like event sequences.
          full_length: If True, the inputs batch will be for the full length of
              each event sequence. If False, the inputs batch will only be for the
              last event of each event sequence. A full-length inputs batch is used
              for the first step of extending the event sequences, since the RNN
              cell state needs to be initialized with the priming sequence. For
              subsequent generation steps, only a last-event inputs batch is used.

        Returns:
          An inputs batch. If `full_length` is True, the shape will be
          [len(event_sequences), len(event_sequences[0]), INPUT_SIZE]. If
          `full_length` is False, the shape will be
          [len(event_sequences), 1, INPUT_SIZE].
        """
        inputs_batch = []
        for events in event_sequences:
            inputs = []
            if full_length:
                for i in range(len(events)):
                    inputs.append(self.events_to_input(events, i))
            else:
                inputs.append(self.events_to_input(events, len(events) - 1))
            inputs_batch.append(inputs)
        return inputs_batch

    def extend_event_sequences(self, event_sequences, softmax):
        """Extends the event sequences by sampling the softmax probabilities.

        Args:
          event_sequences: A list of EventSequence objects.
          softmax: A list of softmax probability vectors. The list of softmaxes
              should be the same length as the list of event sequences.

        Returns:
          A Python list of chosen class indices, one for each event sequence.
        """
        num_classes = len(softmax[0][0])
        chosen_classes = []
        for i in range(len(event_sequences)):
            chosen_class = np.random.choice(num_classes, p=softmax[i][-1])
            event = self.class_index_to_event(chosen_class, event_sequences[i])
            event_sequences[i].append(event)
            chosen_classes.append(chosen_class)
        return chosen_classes

    def evaluate_log_likelihood(self, event_sequences, softmax):
        """Evaluate the log likelihood of multiple event sequences.

        Each event sequence is evaluated from the end. If the size of the
        corresponding softmax vector is 1 less than the number of events, the entire
        event sequence will be evaluated (other than the first event, whose
        distribution is not modeled). If the softmax vector is shorter than this,
        only the events at the end of the sequence will be evaluated.

        Args:
          event_sequences: A list of EventSequence objects.
          softmax: A list of softmax probability vectors. The list of softmaxes
              should be the same length as the list of event sequences.

        Returns:
          A Python list containing the log likelihood of each event sequence.

        Raises:
          ValueError: If one of the event sequences is too long with respect to the
              corresponding softmax vectors.
        """
        all_loglik = []
        for i in range(len(event_sequences)):
            if len(softmax[i]) >= len(event_sequences[i]):
                raise ValueError(
                    'event sequence must be longer than softmax vector (%d events but '
                    'softmax vector has length %d)' % (len(event_sequences[i]),
                                                       len(softmax[i])))
            end_pos = len(event_sequences[i])
            start_pos = end_pos - len(softmax[i])
            loglik = 0.0
            for softmax_pos, position in enumerate(range(start_pos, end_pos)):
                index = self.events_to_label(event_sequences[i], position)
                loglik += np.log(softmax[i][softmax_pos][index])
            all_loglik.append(loglik)
        return all_loglik


class OneHotEventSequenceEncoderDecoder(EventSequenceEncoderDecoder):
    """An EventSequenceEncoderDecoder that produces a one-hot encoding."""

    def __init__(self, one_hot_encoding):
        """Initialize a OneHotEventSequenceEncoderDecoder object.

        Args:
          one_hot_encoding: A OneHotEncoding object that transforms events to and
              from integer indices.
        """
        self._one_hot_encoding = one_hot_encoding

    @property
    def input_size(self):
        return self._one_hot_encoding.num_classes

    @property
    def num_classes(self):
        return self._one_hot_encoding.num_classes

    @property
    def default_event_label(self):
        return self._one_hot_encoding.encode_event(
            self._one_hot_encoding.default_event)

    def events_to_input(self, events, position):
        """Returns the input vector for the given position in the event sequence.

        Returns a one-hot vector for the given position in the event sequence, as
        determined by the one hot encoding.

        Args:
          events: A list-like sequence of events.
          position: An integer event position in the event sequence.

        Returns:
          An input vector, a list of floats.
        """
        input_ = [0.0] * self.input_size
        input_[self._one_hot_encoding.encode_event(events[position])] = 1.0
        return input_

    def events_to_label(self, events, position):
        """Returns the label for the given position in the event sequence.

        Returns the zero-based index value for the given position in the event
        sequence, as determined by the one hot encoding.

        Args:
          events: A list-like sequence of events.
          position: An integer event position in the event sequence.

        Returns:
          A label, an integer.
        """
        return self._one_hot_encoding.encode_event(events[position])

    def class_index_to_event(self, class_index, events):
        """Returns the event for the given class index.

        This is the reverse process of the self.events_to_label method.

        Args:
          class_index: An integer in the range [0, self.num_classes).
          events: A list-like sequence of events. This object is not used in this
              implementation.

        Returns:
          An event value.
        """
        return self._one_hot_encoding.decode_event(class_index)

    def labels_to_num_steps(self, labels):
        """Returns the total number of time steps for a sequence of class labels.

        Args:
          labels: A list-like sequence of integers in the range
              [0, self.num_classes).

        Returns:
          The total number of time steps for the label sequence, as determined by
          the one-hot encoding.
        """
        events = []
        for label in labels:
            events.append(self.class_index_to_event(label, events))
        return sum(self._one_hot_encoding.event_to_num_steps(event)
                   for event in events)
