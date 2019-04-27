# Lucy

### Introduction

Lucy is a smart region proposer for existing Convolutional Neural Network (CNN) architectures, that incorporates soft computing methodologies like Genetic Algorithm (GA) & Particle Swarm Optimization (PSO) to provide better region proposals than conventional selective search algorithm.

### TODOs

- [x] Isolate `GA` class methods from `Regions` class
- [x] Figure out a better architecture for easily changing fitness functions
- [x] Figure out how to handle mutations trying to overgrow original image
- [x] Try out PSO-based region proposer
  - [x] Add fitness function-1
  - [ ] Add fitness function-2
  - [x] Fix the velocity update sanitizer function
- [ ] Write an evaluator script, that runs for whole corpus and integrates with CNN pipeline
- [x] Write a CLI script, that makes it easy to run for a sample image
