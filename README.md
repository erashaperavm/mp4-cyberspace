## mp4-cyberspace

Turn ordinary videos into Cyberpunk-style cyberspace visuals with a pure CPU Python pipeline.

This is my first Vibe Coding experiment.

I normally write Go and Rust, but I wanted to see how far Claude and DeepSeek V4 Pro could go when building a complete computer vision pipeline in Python.

The result is a CPU-only renderer that combines:

- Sparse point-cloud rendering
- Background removal (rembg)
- Pixel sorting
- Datamoshing
- RGB chromatic aberration
- Bloom
- Scanlines
- Motion ghosts
- LUT-based cyberpunk color grading

No GPU is required.

[example.gif](https://github.com/erashaperavm/mp4-cyberspace/blob/main/static/output.gif)

### Features

- ✓ Pure CPU pipeline

- ✓ Automatic foreground extraction

- ✓ Cyberpunk point-cloud renderer

- ✓ Procedural digital wall

- ✓ Dynamic glitch effects

- ✓ Motion ghosting

- ✓ 1080p output

- ✓ Single Python script
