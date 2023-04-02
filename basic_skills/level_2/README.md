# Save a checkpoint

Lightning module automatically saves a checkpoint in the current working directory,
with the state of the last training epoch.

To change the checkpoint path, use the ```default_root_dir``` argument

```python
trainer = Trainer(default_root_dir="some/path")
```

# LightningModule from checkpoint

To load a LightningModule along with its weights and hyperparameters use the
following method

```python
model = MyLightningModule.load_from_checkpoint("/path/to/checkpoint.ckpt")

# disable randomness, dropout, etc ...
model.eval()

# predict with the model
y_hat = model(x)
```
