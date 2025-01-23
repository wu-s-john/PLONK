use ark_ff::FftField;
use ark_poly::{
    univariate::DensePolynomial,
    EvaluationDomain,
    GeneralEvaluationDomain,
    UVPolynomial,
};

/// Converts a slice of evaluations (in a given domain) to a DensePolynomial
/// (coefficient form) by performing an inverse FFT (IFFT).
///
/// # Arguments
/// * `evals` - The polynomial evaluations in the current domain (length must match domain size).
/// * `domain` - The evaluation domain (size must match the length of evals).
///
/// # Returns
/// A DensePolynomial constructed from the coefficient form of the given evaluations.
pub fn evaluations_to_dense_polynomial<F: FftField>(
    evals: &[F],
    domain: &GeneralEvaluationDomain<F>,
) -> DensePolynomial<F> {
    assert_eq!(
        evals.len(),
        domain.size(),
        "Evaluation length must match the domain size"
    );

    // Convert from evaluation form to coefficients by IFFT
    let coeffs = domain.ifft(evals);

    // Create the DensePolynomial directly
    DensePolynomial::from_coefficients_vec(coeffs)
}

/// A helper function that takes a "builder" closure returning a vector of evaluations
/// in the provided domain, then converts them to a DensePolynomial. This design allows
/// for reusable, modular code in scenarios where you need to:
///   1. Construct some polynomial evaluations (e.g. a grand product polynomial)
///   2. Immediately convert them into coefficient form.
///
/// # Arguments
/// * `domain` - The FFT domain for polynomial operations.
/// * `builder` - A function or closure that constructs a Vec<F> of polynomial evaluations of length = domain.size().
///
/// # Returns
/// A DensePolynomial (in coefficient form).
pub fn build_and_convert<F, Builder>(
    domain: &GeneralEvaluationDomain<F>,
    builder: Builder,
) -> DensePolynomial<F>
where
    F: FftField,
    Builder: FnOnce() -> Vec<F>,
{
    let evals = builder();
    assert_eq!(
        domain.size(),
        evals.len(),
        "Domain size and builder evaluations length should match"
    );
    evaluations_to_dense_polynomial(&evals, domain)
} 