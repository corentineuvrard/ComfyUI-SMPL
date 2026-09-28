# models/

Local data only. Nothing in this folder except this README is tracked by git,
and nothing here may be redistributed with the node pack.

```
models/
├── smpl/
│   ├── SMPL_MALE.pkl           # you provide (see README, "Body model files")
│   └── SMPL_FEMALE.pkl
├── smplh/
│   ├── SMPLH_MALE.pkl          # you provide, merged with MANO
│   └── SMPLH_FEMALE.pkl
├── regressor/
│   └── height_weight_to_betas.npz   # generated on first use from your model files
└── segmentation/
    └── smpl_vert_segmentation.json  # downloaded on first use (Meshcapade)
```

Body model files can also live elsewhere: set `SMPL_AVATAR_MODEL_DIR` or use
`<ComfyUI>/models/body_models/` (same `smpl/` and `smplh/` layout).
