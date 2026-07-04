# Examples

Generate a local synthetic scene:

```bash
bananavision synthetic --image examples/synthetic_banana_scene.jpg --truth examples/synthetic_banana_scene.truth.json
bananavision infer examples/synthetic_banana_scene.jpg --output runs/infer --config configs/banana_uav.yaml
```

The synthetic generator is for pipeline testing only. It is not a substitute for field validation.
