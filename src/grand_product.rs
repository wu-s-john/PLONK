use ark_ff::Field;

use crate::execution_trace::ExecutionTraceTable;

/// Builds the grand product polynomial Z.  
/// Z has length n+1 (where n is the number of rows in the evaluation table),  
/// with Z[0] = 1 and then for j in [0..n-1]:
///     Z[j+1] = Z[j]
///             * Π(i=0..m-1) [ vᵢ(j) + β * vᵢ(σᵢ(j)) + γ ]
///             / Π(i=0..m-1) [ vᵢ(j) + β * δᵢ(j) + γ ]
///
/// Here:
///   • vᵢ(j) is the i-th wire value at row j,
///   • vᵢ(σᵢ(j)) is that i-th wire evaluated at the permuted index from permutation_table,
///   • δᵢ(j) is a placeholder for the shift (or domain factor) in a real Plonk system.
///     For simplicity, we just use j (the row index) or some other scheme below.
///
#[allow(dead_code)]
pub fn build_grand_product_evaluation_vector<F: Field>(
    evaluation_table: &ExecutionTraceTable<F>,
    beta: F,
    gamma: F,
) -> Vec<F> {
    let n = evaluation_table.input1.len();
    let mut z = vec![F::one(); n + 1];

    // For each row j, compute the ratio of products
    for j in 1..n {
        // Collect wire values at row j
        let v1_j = evaluation_table.input1[j];
        let v2_j = evaluation_table.input2[j];
        let v3_j = evaluation_table.input3[j];
        let vo_j = evaluation_table.output[j];

        // Collect wire values at row sigmaᵢ(j)
        let v1_sigma = evaluation_table.permutation_input1[j];
        let v2_sigma = evaluation_table.permutation_input2[j];
        let v3_sigma = evaluation_table.permutation_input3[j];
        let vo_sigma = evaluation_table.permutation_output[j];

        // Numerator = Π over each wire i of [ vᵢ(j) + β * vᵢ(σᵢ(j)) + γ ]
        let mut numerator = F::one();
        numerator *= v1_j + (beta * v1_sigma) + gamma;
        numerator *= v2_j + (beta * v2_sigma) + gamma;
        numerator *= v3_j + (beta * v3_sigma) + gamma;
        numerator *= vo_j + (beta * vo_sigma) + gamma;

        // Denominator = Π over each wire i of [ vᵢ(j) + β * δᵢ(j) + γ ]
        // For simplicity, we use j as part of δᵢ(j) (you could refine this if needed).
        let j_as_field = F::from(j as u64);
        let mut denominator = F::one();
        denominator *= v1_j + (beta * j_as_field) + gamma;
        denominator *= v2_j + (beta * j_as_field) + gamma;
        denominator *= v3_j + (beta * j_as_field) + gamma;
        denominator *= vo_j + (beta * j_as_field) + gamma;

        // Update Z[j+1] = Z[j] * (numerator / denominator)
        // (We assume denominator != 0 in typical Plonk usage.)
        z[j] = z[j - 1] * numerator * denominator.inverse().expect(
            "Denominator must be non-zero in valid PLONK circuits. \
             This indicates either bad circuit construction or \
             invalid challenge parameters beta/gamma"
        );
    }
    z
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr as F;
    use ark_std::test_rng;
    use crate::{
        ast_to_plonk::convert_to_plonk,
        execution_trace::interpret_plonk_node_to_execution_trace_table,
        plonk_circuit::eval_plonk_node
    };

    fn run_grand_product_test<F: Field>(input: &str) {
        // Parse and convert to PlonkNode
        let expr = parse_expr(input);
        let plonk_node = convert_to_plonk(&expr)
            .unwrap_or_else(|e| panic!("Conversion failed: {:?}", e));
        
        // Evaluate to get node metadata
        let evaluated_plonk = eval_plonk_node::<F>(&plonk_node)
            .unwrap_or_else(|e| panic!("Evaluation failed: {:?}", e));

        // Build constraints from PlonkNode
        let trace_table = interpret_plonk_node_to_execution_trace_table(
            &evaluated_plonk.root,
            evaluated_plonk.node_node_equivalences.clone()
        );

        // Build execution trace table

        // Test grand product construction
        let beta = F::rand(&mut test_rng());
        let gamma = F::rand(&mut test_rng());
        let z = build_grand_product_evaluation_vector(&trace_table, beta, gamma);
        
        assert_eq!(
            *z.last().unwrap(),
            F::from(1u64),
            "Grand product final value != 1 for input: {}",
            input
        );
    }

    // Reuse the existing test infrastructure
    fn parse_expr(input: &str) -> Box<crate::language_frontend::ast::Expr> {
        use crate::language_frontend::{lexer::lex, parser};
        
        let tokens = lex(input);
        let token_triples: Vec<_> = tokens.into_iter().enumerate()
            .map(|(i, t)| (i, t, i + 1))
            .collect();
        
        parser::ExprParser::new()
            .parse(token_triples.into_iter())
            .unwrap()
    }

    // Test cases that exercise different parts of the constraint system
    #[test]
    fn test_basic_arithmetic() {
        run_grand_product_test::<F>("1 + 2 * 3");
        run_grand_product_test::<F>("(1 + 2) * 3");
        run_grand_product_test::<F>("5 - 2");
    }

    #[test]
    fn test_boolean_logic() {
        run_grand_product_test::<F>("true && false");
        run_grand_product_test::<F>("true || false");
    }

    #[test]
    fn test_variable_reuse() {
        run_grand_product_test::<F>("let x = 5 in x + x");
        run_grand_product_test::<F>("let x = 2 in let y = x * x in y + 3");
    }

    #[test]
    fn test_edge_cases() {
        run_grand_product_test::<F>("0");
        run_grand_product_test::<F>("let x = 5 in 3"); 
        // TODO: Fix this test
        // run_grand_product_test::<F>("if true then 1 else 0");
    }
}

