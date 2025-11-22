# cuGraph Playground

## Installation (didn't work, latest rapidsai on docker is 23.06, while Sirius uses 25.10)

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


## RAPIDS Installation with Conda

Set up conda env
```bash
conda create -n rapids-25.10 -y
conda activate rapids-25.10
conda install -c rapidsai -c conda-forge -c nvidia \
    rapidsai::libcudf=25.10 \
    rapidsai::cudf=25.10 \
    rapidsai::cugraph=25.10
```

To start conda env (if env already setup)
```bash
conda activate rapids-25.10
```

## DuckDB v1.2.1 Installation with Conda (to work with Substrait) 

```bash
conda activate rapids-25.10
conda install -c conda-forge duckdb=1.2.1
```



