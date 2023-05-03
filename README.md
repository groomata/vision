[그루마타 온라인 클래스](https://www.groomata.com)에서 진행한 프로젝트입니다.

프로젝트는 `pytorch-lightning`을 사용하여 구성되었으며, 모든 실험 설정과 결과물은 `wandb`에 자동으로 기록됩니다. `wandb` workspace의 예시는 [여기](https://wandb.ai/groomata-vision/groovis/groups/simclr/workspace?workspace=user-groomata-vision)에서 확인할 수 있습니다.

실험 설정은 `hydra`에 의해 관리되기에 *CLI*를 통해 제어할 수 있습니다. 설정은 `hydra-zen`을 통해 자동으로, *programmatic*하게 생성됩니다.


### Usage example

*Training*은 `docker`를 이용하는 것이 권장됩니다. `groomata/vision` 이미지를 이용하면 되며, 상기했듯 모든 실험 설정은 *CLI*를 이용해 제어할 수 있습니다.

```sh
docker run \
    --gpus=all \
    --ipc=host \
    --volume=/path/to/volume:/vision/.cache \
    --env-file=/path/to/.env \
    --tty \
    groomata/vision \
    # 원하는 실험 설정으로 덮어 쓰면 됩니다.
    optimizer.lr=0.0001 \
    datamodule.dataloader.batch_size=64 \
    trainer.max_epochs=100 \
    trainer.gradient_clip_algorithm="norm" \
    trainer.gradient_clip_val=1.0
```

### Note

프로젝트를 작성해나가는 과정은 [그루마타 온라인 클래스](https://www.groomata.com)에서 *end-to-end*로 경험할 수 있어요.
