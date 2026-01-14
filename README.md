# Neuro-Evolution Sandbox

An experimental research sandbox for training and composing neural network based control policies inside a 2D physics driven world.

This project explores how reusable low level control skills can be learned, evaluated, and combined using fast parallel simulation, clean control abstractions, and a mixture of experts architecture. The codebase is intentionally simple, explicit, and heavily commented, with an emphasis on clarity of mechanics rather than framework complexity.

---

## High Level Idea

Creatures are modeled as thrust and rotation driven ships operating under impulse physics in a continuous 2D world. Each creature is controlled by a **Controller** abstraction that converts observations into low level action signals.

The long term goal is to:
- Train many small neural networks as **specialist skills**
- Freeze those specialists once learned
- Compose them using learned **gating networks**
- Produce smooth, fuzzy transitions between behaviors rather than hard state machines

This allows complex behavior to emerge from simple, well defined components.

---

## Core Concepts

### Creatures
- Represented as lightweight physics bodies
- Move via forward thrust and rotational torque
- Use continuous impulse based physics rather than grid or tile logic
- Wrap around a toroidal world (Asteroids style)

### Controllers
- Abstract interface responsible for producing control signals
- Current implementation includes a keyboard controller
- Future implementations include:
  - Neural network controllers
  - Gating controllers
  - Scripted or hybrid controllers

Creatures do not know *how* they are controlled.

### World
- Continuous 2D space
- Toroidal topology (edges wrap)
- World size dynamically matches the window size
- Designed to scale to many parallel simulated worlds later

### Physics
- Fixed timestep integration
- Explicit linear and angular damping
- Speed clamping for stability and learnability
- Torch tensors used for future GPU and batch simulation compatibility

---

## Why This Exists

This project is not about building a game.

It exists to explore:
- How low level motor skills can be learned independently
- How behaviors can be composed without brittle state machines
- How neural policies behave under clean, deterministic physics
- How to scale training across many simulated environments

The emphasis is on **control**, **composition**, and **emergence**, not visuals or content.

---

## Current State

- Single creature simulation
- Keyboard driven controller
- Resizable window with toroidal world wrapping
- Clean separation between physics, control, and rendering
- Extensive inline documentation

The code is intentionally kept simple and explicit as a foundation for future experimentation.

---

## Planned Directions

- Sensor systems (goal vectors, obstacle rays, threat awareness)
- Batched simulation of many worlds in parallel
- Neural network controllers trained via fitness based evaluation
- Mixture of experts and learned gating networks
- Headless high speed training modes
- Replay and inspection tools

---

## Philosophy

This project favors:
- Explicit math over hidden engine behavior
- Simple abstractions over deep inheritance
- Learnable control spaces over hand tuned logic
- Systems that scale naturally to learning

It is as much a research notebook as it is a codebase.

---

## Status

This is an evolving experimental project.

Expect:
- Refactors
- Abandoned ideas
- Exploratory code paths

Stability is secondary to understanding.

---

## License

MIT License

Use, modify, and experiment freely.
