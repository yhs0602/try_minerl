# How to install minerl on MacOS

- Must use Jdk 8 (e.g. adoptopenjdk8)
- https://github.com/minerllabs/minerl/issues/659#issuecomment-1306635414

# Performance comparison
> On M1 Pro (640 x 360 resolution, random agent)
- [MineRL](https://github.com/minerllabs/minerl): 27~32 FPS(=TPS)
- [Craftground](https://github.com/yhs0602/CraftGround): 100~ FPS(=TPS)
- [Crafter(2D)](https://github.com/danijar/crafter): 539~ FPS(=TPS)

* Had to fix crafter `render()` for latest gym