# Applied-Mathematics with JAX 

Foundations of Applied Mathematics

## Open port on docker container

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm --shm-size=8gb -p 7070:7070 vsc-cuda-container-92f8d0bcb7ce38ae6a144f94ac61d015
```

## Install Jupyter Notebook

```
pip3 install notebook
```

## Run Jupyter insider the container

```
jupyter notebook --ip 0.0.0.0 --port=7070 --no-browser --allow-root
``` 

## Connect jupyter server in VS Code