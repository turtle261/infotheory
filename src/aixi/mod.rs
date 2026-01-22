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
//! ## VM Backends
//!
//! Two VM environment backends are available:
//!
//! - **vm**: Original libvirt-based implementation (requires libvirt/QEMU)
//! - **vm_nyx**: High-performance Firecracker-based implementation using nyx-lite
//!   (10,000+ resets/second, requires KVM)

pub mod agent;
pub mod common;
pub mod environment;
pub mod mcts;
pub mod model;
#[cfg(feature = "vm")]
pub mod vm;
#[cfg(feature = "vm")]
pub mod vm_nyx;
