<h1 align="center">PyTorch Project Template</h1>

This is an example of a clean, reproducible, and boilerplate-free deep learning project
that was developed as part of the [Groomata deep learning course](https://www.groomata.com).

This project is organized using `pytorch-lightning`, and all configurations and artifacts can be uploaded to `wandb` without any compromise. You can see an example wandb workspace [here](https://wandb.ai/groomata-vision/groovis/groups/simclr/workspace?workspace=user-groomata-vision). All configurations are programmatically generated and maintained by `hydra` and `hydra-zen`.

### Usage example

```sh
docker run \
    --gpus=all \
    --ipc=host \
    --volume=/path/to/volume:/vision/.cache \
    --env-file=/path/to/.env \
    --tty \
    groomata/vision \
    # Override any configurations you want
    optimizer.lr=0.0001 \
    datamodule.dataloader.batch_size=64 \
    trainer.max_epochs=100 \
    trainer.gradient_clip_algorithm="norm" \
    trainer.gradient_clip_val=1.0
```
