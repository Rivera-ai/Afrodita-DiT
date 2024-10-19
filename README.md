## I am implementing conditional DiT by text, but I still have errors and an example of these errors is:

New error: 

  ```
⚡ ~ python TrainDiTTest.py --batch-size 4 --epochs 10
[2024-10-19 22:12:36] Experiment directory created at results/025-DiT-S-8
Pesos text_projector: torch.Size([384, 768]), Bias: torch.Size([384])
[2024-10-19 22:12:36] DiT Parameters: 4,183,808
Using the latest cached version of the dataset since tungdop2/nsfw_w_caption couldn't be found on the Hugging Face Hub
[2024-10-19 22:12:36] Using the latest cached version of the dataset since tungdop2/nsfw_w_caption couldn't be found on the Hugging Face Hub
Found the latest cached dataset configuration 'default' at /home/zeus/.cache/huggingface/datasets/tungdop2___nsfw_w_caption/default/0.0.0/b3c4989a2ec9d32a80e22e7a3b77c6748291d205 (last modified on Sat Oct 19 21:39:14 2024).
[2024-10-19 22:12:36] Found the latest cached dataset configuration 'default' at /home/zeus/.cache/huggingface/datasets/tungdop2___nsfw_w_caption/default/0.0.0/b3c4989a2ec9d32a80e22e7a3b77c6748291d205 (last modified on Sat Oct 19 21:39:14 2024).
[2024-10-19 22:12:36] Dataset contains 4,289 images
[2024-10-19 22:12:36] Starting epoch 0
Image size after transform: torch.Size([4, 3, 256, 256])
Labels: tensor([[-9.4103e-02, -9.0301e-02,  1.0699e-01,  ...,  7.6761e-02,
          3.6062e-01, -5.2831e-02],
        [-1.7130e-01, -6.4920e-03,  4.5896e-02,  ...,  1.7172e-01,
          3.6261e-01,  1.0613e-04],
        [-2.9661e-02, -8.0374e-03,  1.2952e-01,  ...,  1.6787e-01,
          1.6331e-01,  3.7903e-03],
        [-1.9565e-01, -9.9726e-03,  9.9829e-02,  ...,  3.6985e-02,
          2.7002e-01,  9.0892e-02]], device='cuda:0')
Max label index: 0.8368053436279297
Min label index: -7.4886345863342285
labels.shape: torch.Size([4, 768])
t.shape: torch.Size([4, 384]), y.shape: torch.Size([4, 768, 384]), text_embed.shape: torch.Size([4, 384])
DiTBlock input shapes:
x shape: torch.Size([4, 16, 384])
c shape: torch.Size([4, 768, 384])
adaLN_modulation output shape: torch.Size([4, 768, 2304])
After chunk:
shift_msa shape: torch.Size([4, 768, 384])
scale_msa shape: torch.Size([4, 768, 384])
gate_msa shape: torch.Size([4, 768, 384])
Traceback (most recent call last):
  File "/teamspace/studios/this_studio/TrainDiTTest.py", line 212, in <module>
    main(args)
  File "/teamspace/studios/this_studio/TrainDiTTest.py", line 174, in main
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
  File "/teamspace/studios/this_studio/ModelDiT.py", line 246, in forward
    x = self.unpatchify(x)                   # (N, out_channels, H, W)
  File "/teamspace/studios/this_studio/ModelDiT.py", line 211, in unpatchify
    assert h * w == x.shape[1], f"h: {h}, w: {w}, x.shape[1]: {x.shape[1]}"
AssertionError: h: 27, w: 27, x.shape[1]: 768
```

## I have based myself on the Meta repo: https://github.com/facebookresearch/DiT.git
