# cuGraph Playground

## Installation

Pull image for prebuilt cuda env if you haven't
```bash
docker pull rapidsai/rapidsai:23.06-cuda11.8-runtime-ubuntu22.04-py3.10
```

Start the image
```bash
docker run --gpus all --rm -it \
  -v /home/andy/sirius-spatial-playground:/workspace \
  -w /workspace \
  rapidsai/rapidsai:23.06-cuda11.8-runtime-ubuntu22.04-py3.10 \
  bash
```