// This module is responsible for building the polynomial constraints for the PLONK circuit.

// Tests for permutation constraint polynomial
// - We need to make sure that the permutation constraint polynomial is all 0 at the points of evaluation
// ((a(X) + βX + γ)(b(X) + βk1X + γ)(c(X) + βk2X + γ)z(X)) - (a(X) + βSσ1 (X) + γ)(b(X) + βSσ2 (X) + γ)(c(X) + βSσ3 (X) + γ)z(Xω))
//

// Tests for the gate constraints polynomial
// Int -> (input_1 - output) * int_selector
// Bool -> (input_1 - output) * bool_selector
// Add -> (input_1 + input_2 - output) * add_selector
// Sub -> (input_1 - input_2 - output) * sub_selector
// Mult -> (input_1 * input_2 - output) * mult_selector
// Div -> (input_1 / input_2 - output) * div_selector
// Eq -> (input_1 == input_2 - output) * eq_selector
// Not -> (input_1 - !input_2) * not_selector
// If -> (input_1 - output) * if_selector
// Let -> (input_1 - output) * let_selector

// TODO: We need to make simple tests not on the evaluated domain to see everything working

// Re-export from execution_trace_polynomial
pub use crate::execution_trace_polynomial::{
    ExecutionTrace, 
    build_execution_trace_polynomial,
    build_execution_trace_extended_evaluation_form,
};

// Re-export from polynomial_constraints_direct
pub use crate::polynomial_constraints_direct::{
    build_permutation_constraint_polynomial_direct as build_permutation_constraint_polynomial_no_fft,
    build_gate_constraint_polynomial,
};

// Re-export from polynomial_constraints_fft
pub use crate::polynomial_constraints_fft::{
    build_permutation_constraint_polynomial_fft as build_permutation_constraint_polynomial,
};
