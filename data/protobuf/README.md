## What is this and why do I need them

Their protobuf files, you need to install `protoc` (`google-protobuf` on Linux), it generates the classes necessary for storing midi data.

Run these once you have `protoc` installed:

`protoc music.proto --python_out=.`

`protoc generator.proto --python_out=.`

Python files should show up here, `generator_pb2.py` and `music_pb2.py`. Once you see them your set. 