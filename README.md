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
    - [ ] Implement blinding of the wire polynomials


- We have to organize the code better



## Building the Prover process of Plonk
### Setup
Compute SRS Strings {g, g^s, g^s^2, ..., g^s^n}
- Setup execution trace table:
  - This includes the rows of each operation in the table
  - This also includes the equality constraint of each variable
- Setup Selector polynomial `s(X)`  via FFTs
  - We need to 2 the execution trace from the lambda calculus circuit and then we construct the selector polynomial from the operation in each row of the execution trace
  - **How to test**
    - Given that we have the execution trace, we need to make sure that the node for the selector polynomial is correct
- Setup permutation polynomial `𝜎_𝑖(𝑋)`  via FFTs
  - **How to test**
    - We have a cycle for just two points and make sure that the cycle is correct
    - We have a cycle for 3 points and make sure that the cycle is correct
    - We have a cycle for 4 points and make sure that the cycle is correct
    - If there is an input that is not connected, then we need it's just it's own cycle
    - The permutation polynomial is bijective to the identity permutation
    - For a sample plonk circuit, can we test that the permutation polynomial is correct?
- Compute the KZG commitment of the selector polynomial `s(X)`


- So, we can just have one giant execution table that includes what operation was executed for each row
- Then, we can have each expression be tied to a expression id and then connect them together. 
- Each expression and variable is tied to a node id.
    - Whenever you use the expression or variable, you reference the node id
    - expression ids can be tied to each other as well. 
    - You can cell ids be equivalent to expression ids. Then, we want to make equivalent groupings of each cell and expression ids.



## The Actual Proving Process
- Derive the wire polynomials `w(X)`
  - Tests
    - Given that we have the execution trace, we need to make sure that the node for the wire polynomials is correct. Like the operations would equal to the wire polynomials
- Produce the grand product polynomial `Z(𝑋)` via FFTs
  - First sample permutation challenges (β, γ) from verifier
  - Then compute the grand product polynomial `Z(𝑋)`
  - We need to compute KZG commitment of `Z(X)` 
    - [z]1
  - The grand product polynomial `Z(X)` equals to 1 in the end
- **How to test**
  - We need to test that we computed `Z(x)` correctly
    - In the end, we need to make sure that the grand product polynomial `Z(X)` equals to 1



- Produce the aggregation polynomial `agg(𝑋)`  
  - Sample α which will be used to combine the constraint polynomials
  - Compute wire constraint polynomial
     - **How to test**
        - For each row, ensure that the wire constraint polynomial equals to zero
  - Compute permutation constraint polynomial
        - For each row, ensure that the permutation constraint polynomial equals to zero
- Compute the polynomial quotient polynomial `q(t)` where you divide on the vanishing polynomial `Z_H(X)`
- **How to test**
  - It would be good to test that the quotient polynomial is correct by checking that the quotient polynomial multiplied by the vanishing polynomial equals to the grand product polynomial evaluating the polynomial at a random point
- Split the quotient polynomial into 4 polynomials
- Compute the KZG commitments of the 4 polynomials
  - [t_0(X)]1
  - [t_1(X)]1
  - [t_2(X)]1
  - [t_3(X)]1


- Now do opening challenges for the polynomials, w(x), s(x), z(x)
  - Verifer samples challenges for the base polynomials, ε
  - Compute the opening of the polynomials -> w(ε), s(ε), z(ε), z(ω ε)



- Now do a batch commitment to prove the opening of the polynomials as well as the base commitment
  - Verifier samples challenges for the base polynomials
  - Compute the linearization polynomial
     - *NOTE*: The linearization polynomial can be seen as a bivariate polynomial
     - At the point of evaluation, we need to make sure that the linearization polynomial equals to the underlying `agg(X)` polynomial
  - Compute the KZG commitment of the linearization polynomial
  - Compute the KZG proof of the linearization polynomial


- Verifier supplies challenge for quotient polynomial α
- Compute quotient polynomial `𝑡(𝑋)` with that challenge
- TODO: for verification purposes, we can test piece by piece



