from zonos.model import Zonos
from zonos.utils import DEFAULT_DEVICE as device

model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)

print("Model config:")
print(model.config)

print("DAC config:")
print(model.autoencoder.dac.config)

print("Decoder layers:")
print(model.autoencoder.dac.decoder)