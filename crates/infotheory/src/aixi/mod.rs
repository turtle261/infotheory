//! AIXI Agent Implementations
//!
//! This module contains:
//! - Monte Carlo AIXI (MC-AIXI) with pluggable predictive models.
//! - AIQI from "A Model-Free Universal AI" with phase-indexed return prediction.
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
pub mod aiqi;
pub mod common;
pub mod environment;
pub mod mcts;
pub mod model;
pub(crate) mod planner_spec;
#[cfg(feature = "vm")]
pub mod vm_nyx;
