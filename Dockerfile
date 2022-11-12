FROM nvidia/cuda:11.6.0-cudnn8-runtime-ubuntu20.04

WORKDIR /vision

ENV DEBIAN_FRONTEND noninteractive \
    TZ Asia/Seoul

RUN apt-get update -y && apt-get install -y \
    python3.9 python3.9-dev python3.9-venv

RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.1.14

ENV PATH /root/.local/bin:$PATH

COPY . .

RUN poetry install --no-dev --no-interaction && \
    poetry run pip install \
    torch==1.12.1+cu116 torchvision==0.13.1+cu116 \
    -f https://download.pytorch.org/whl/torch_stable.html

ENTRYPOINT ["poetry", "run", "python", "train.py"]
