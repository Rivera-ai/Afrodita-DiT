## I am implementing conditional DiT by text, but I still have errors and an example of these errors is:

New error: 

  ```
⚡ ~ python TrainDiTTest2.py --batch-size 4 --epochs 10
[2024-10-21 01:22:04] Experiment directory created at results/025-DiT-S-8
[2024-10-21 01:22:05] DiT Parameters: 49,193,600
Using the latest cached version of the dataset since tungdop2/nsfw_w_caption couldn't be found on the Hugging Face Hub
[2024-10-21 01:22:05] Using the latest cached version of the dataset since tungdop2/nsfw_w_caption couldn't be found on the Hugging Face Hub
Found the latest cached dataset configuration 'default' at /home/zeus/.cache/huggingface/datasets/tungdop2___nsfw_w_caption/default/0.0.0/b3c4989a2ec9d32a80e22e7a3b77c6748291d205 (last modified on Mon Oct 21 00:43:56 2024).
[2024-10-21 01:22:05] Found the latest cached dataset configuration 'default' at /home/zeus/.cache/huggingface/datasets/tungdop2___nsfw_w_caption/default/0.0.0/b3c4989a2ec9d32a80e22e7a3b77c6748291d205 (last modified on Mon Oct 21 00:43:56 2024).
[2024-10-21 01:22:05] Dataset contains 4,289 images
[2024-10-21 01:22:05] Starting epoch 0
shift_msa: torch.Size([4, 384]), scale_msa: torch.Size([4, 384]), shift_cross: torch.Size([4, 384]), scale_cross: torch.Size([4, 384]), shift_mlp: torch.Size([4, 384]), scale_
mlp: torch.Size([4, 384])
torch.Size([4, 384])
Traceback (most recent call last):
  File "/teamspace/studios/this_studio/TrainDiTTest2.py", line 188, in <module>
    main(args)
  File "/teamspace/studios/this_studio/TrainDiTTest2.py", line 148, in main
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
  File "/teamspace/studios/this_studio/ModelDiT2.py", line 233, in forward
    x = block(x, c, text_embed)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/teamspace/studios/this_studio/ModelDiT2.py", line 137, in forward
    x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
RuntimeError: The size of tensor a (4) must match the size of tensor b (16) at non-singleton dimension 1
```

# Train File test with MNIST is ``` TrainDiT2TestMnist.py ```


## I have based myself on the Meta repo: https://github.com/facebookresearch/DiT.git
