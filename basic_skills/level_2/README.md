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

## Save hyperparameters

The LightningModule allows us to automatically save all the hyperparameters
passed to ```init``` by calling ```self.save_hyperparameters()```.

```python
class MyLightningModule(LightningModule):
    def __init__(self, learning_rate, another_parameter, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
```

The hyperparameters are saved to the "hyper_parameters" key in the checkpoint.

```python
checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
print(checkpoint["hyper_parameters"])
# output: {"learning_rate": the_value, "another_parameter": the other value}
```

The LightningModule also has access to the Hyperparameters

```python
model = MyLightningModule.load_from_checkpoint("/path/to/checkpoint.ckpt")
print(model.learning_rate)
```

# Initialize with other parameters

If we used the ```self.save_hyperparameters()``` method in the ```init``` of the
LightningModule, we could initialize the model with different hyperparameters.

```python
LitModel(in_dim=32, out_dim=10)

model = LitModel.load_from_checkpoint(PATH)  # uses in_dim=32, out_dim=10

model = LitModel.load_from_checkpoint(PATH, in_dim=128, out_dim=10)  # uses in_dim=128, out_dim=10
```

# nn.Module from checkpoint

Lightning checkpoints are fully compatible with plain torch nn.Module

```python
checkpoint = torch.load(CKPT_PATH)
print(checkpoint.keys())
```

For example,

```python
class Encoder(nn.Module):
    ...

class Decoder(nn.Module):
    ...

class Autoencoder(pl.LightningModule):
    def __init__(self, encoder, decoder, *args, **kwargs):
        ...
autoencoder = Autoencoder(Encoder(), Decoder())
```

Once the autoencoder has trained, pull out the relevant weights for the torch nn.Module

```python
checkpoint = torch.load(CKPT_PATH)
encoder_weights = checkpoint["encoder"]
decoder_weights = checkpoint["decoder"]
```

# Disable checkpoiting

```python
trainer = Trainer(enable_checkpointing=False)
```

# Resume training state

```python
model = LitModel()
trainer = Trainer()
# automatically restores model, epoch, step, LR schedulers, etc. 
trainer.fit(model, ckpt_path="path/to/my_checkpoint.ckpt")
```
