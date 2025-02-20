// Data structures
pub mod union_find;

// Core modules
pub mod plonk_circuit;
pub mod execution_trace;
pub mod offset_table;
pub mod permutation_polynomial;
pub mod polynomial_utils;
pub mod ast_to_plonk;

// Language frontend module
pub mod language_frontend;

// Re-export commonly used types
pub use union_find::UnionFind;

// ... existing code ... 