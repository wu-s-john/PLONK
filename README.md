# PLONK Compiler

This project is a compiler for the PLONK proof system. It takes a lambda-based circuit and compiles it into a PLONK proof system.

## CURRENT STATUS

- [ ] Implement the frontend Lambda Calculus interface
    - [ ] Implement the lambda calculus parser
    - [ ] Implement the lambda calculus type checker
- [ ] Implement the backend PLONK proof system
    - [ ] Implement the PLONK prover
      - [X] Implement the PLONK Gate Constraints Polynomial
      - [X] Implement the PLONK Permutation Grand Product Polynomial
      - [X] Implement the PLONK Permutation Constraint Polynomial
      - [ ] Implement commitment scheme
        - [ ] Implement the PLONK Commitment for the wire polynomials and the permuation polynomial
        - [ ] Implement the batched opening of the commitments
    - [ ] Implement the PLONK verifier
        - [ ] Verify KZG proofs
        - [ ] Implement the PLONK Verifier constraints
    - [ ] Implement the verifier


