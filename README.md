# How to install minerl on MacOS

- Must use Jdk 8 (e.g. adoptopenjdk8)
- https://github.com/minerllabs/minerl/issues/659#issuecomment-1306635414

# Performance comparison
> On M1 Pro (640 x 360 resolution, random agent)
- MineRL: 27~32 FPS(=TPS)
- Craftground: 100~ FPS(=TPS)
- Crafter(2D): 539~ FPS(=TPS)

* Had to fix crafter `render()` for latest gym