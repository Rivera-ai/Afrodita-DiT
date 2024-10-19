## I am implementing conditional DiT by text, but I still have errors and an example of these errors is:

  ```
⚡ ~ python TrainDiTTest.py --batch-size 4 --epochs 10                  
[2024-10-19 20:41:03] Experiment directory created at results/025-DiT-S-8
Pesos text_projector: torch.Size([384, 768]), Bias: torch.Size([384])
[2024-10-19 20:41:03] DiT Parameters: 5,364,992
Using the latest cached version of the dataset since tungdop2/nsfw_w_caption couldn't be found on the Hugging Face Hub
[2024-10-19 20:41:03] Using the latest cached version of the dataset since tungdop2/nsfw_w_caption couldn't be found on the Hugging Face Hub
Found the latest cached dataset configuration 'default' at /home/zeus/.cache/huggingface/datasets/tungdop2___nsfw_w_caption/default/0.0.0/b3c4989a2ec9d32a80e22e7a3b77c6748291d205 (last modified on Sat Oct 19 20:06:46 2024).
[2024-10-19 20:41:03] Found the latest cached dataset configuration 'default' at /home/zeus/.cache/huggingface/datasets/tungdop2___nsfw_w_caption/default/0.0.0/b3c4989a2ec9d32a80e22e7a3b77c6748291d205 (last modified on Sat Oct 19 20:06:46 2024).
[2024-10-19 20:41:03] Dataset contains 4,289 images
[2024-10-19 20:41:03] Starting epoch 0
Labels: tensor([[-0.0364, -0.0176,  0.1181,  ..., -0.1646,  0.1532,  0.0633],
        [-0.1592, -0.0077,  0.0200,  ...,  0.1351,  0.2982,  0.1474],
        [-0.1600, -0.0072,  0.0421,  ...,  0.0402,  0.2500,  0.0305],
        [-0.1266,  0.0537,  0.0225,  ...,  0.0448,  0.4064, -0.0006]])
Max label index: 0.7948185205459595
Min label index: -7.7718071937561035
labels.shape: torch.Size([4, 768])
t.shape: torch.Size([4, 384]), y.shape: torch.Size([4, 768, 384]), text_embed.shape: torch.Size([4, 384])
Traceback (most recent call last):
  File "/teamspace/studios/this_studio/TrainDiTTest.py", line 211, in <module>
    main(args)
  File "/teamspace/studios/this_studio/TrainDiTTest.py", line 173, in main
    loss_dict = diffusion.training_losses(model, x, t, model_kwargs=model_kwargs)
  File "/teamspace/studios/this_studio/ReSpace.py", line 88, in training_losses
    return super().training_losses(self._wrap_model(model), *args, **kwargs)
  File "/teamspace/studios/this_studio/GaussianDiffusion.py", line 737, in training_losses
    model_output = model(x_t, t, **model_kwargs)
  File "/teamspace/studios/this_studio/ReSpace.py", line 116, in __call__
    return self.model(x, new_ts, **kwargs)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/teamspace/studios/this_studio/ModelDiT.py", line 220, in forward
    x = block(x, c)                      # (N, T, D)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/teamspace/studios/this_studio/ModelDiT.py", line 95, in forward
    x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
  File "/teamspace/studios/this_studio/ModelDiT.py", line 8, in modulate
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
RuntimeError: The size of tensor a (384) must match the size of tensor b (2304) at non-singleton dimension 3
```

## I have based myself on the Meta repo: https://github.com/facebookresearch/DiT.git
