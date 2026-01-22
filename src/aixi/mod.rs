//! MC-AIXI Implementation
//!
//! This module contains an implementation of the Monte Carlo AIXI algorithm
//! using various predictive models (CTW, ROSA, RWKV) as backends.
//!
//! AIXI is a theoretical mathematical formalism for universal artificial intelligence,
//! which combines Solomonoff induction with sequential decision theory.
//! This implementation follows the "Monte Carlo" approach (MC-AIXI) introduced by
//! Veness et al., which uses Monte Carlo Tree Search (MCTS) to approximate
//! the optimal policy.
//!
//! ## VM Backend
//!
//! A high-performance Firecracker-based VM environment is available via the `vm` feature:
//!
//! - **NyxVmEnvironment**: Uses nyx-lite for 10,000+ resets/second (requires KVM).


pub mod agent;
pub mod common;
pub mod environment;
pub mod mcts;
pub mod model;
#[cfg(feature = "vm")]
pub mod vm_nyx;
