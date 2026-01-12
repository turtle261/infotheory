//! MC-AIXI-CTW Implementation
//!
//! This module contains an implementation of the Monte Carlo AIXI algorithm
//! using various predictive models (CTW, ROSA, RWKV) as backends.
//!
//! AIXI is a theoretical mathematical formalism for universal artificial intelligence,
//! which combines Solomonoff induction with sequential decision theory.
//! This implementation follows the "Monte Carlo" approach (MC-AIXI) introduced by
//! Veness et al., which uses Monte Carlo Tree Search (MCTS) to approximate 
//! the optimal policy.

pub mod model;
pub mod mcts;
pub mod agent;
pub mod environment;
pub mod common;
