from torchnet import meter
import torch as t

loss_meter = meter.AverageValueMeter()
loss_meter.reset()
for i in range(10):
    tensor = t.Tensor([i])
    num = tensor.data.item()
    loss_meter.add(num)
print(loss_meter.value())
