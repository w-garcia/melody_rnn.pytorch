class BadTimeSignatureException(Exception):
  pass


class MultipleTimeSignatureException(Exception):
  pass


class MultipleTempoException(Exception):
  pass


class NegativeTimeException(Exception):
  pass


class QuantizationStatusException(Exception):
  """Exception for when a sequence was unexpectedly quantized or unquantized.

  Should not happen during normal operation and likely indicates a programming
  error.
  """
  pass