### Definition of a Hybrid Automaton

A hybrid automaton is a widely adopted formalism to characterize discrete-continuous behaviors of hybrid systems. It extends finite-state machines by associating each discrete location with a vector of continuously evolving dynamics, subject to discrete jumps upon transitions.

An n-dimensional hybrid automaton is defined as a sextuple:
**H** ≜ (Q, V, F, Inv, Init, T)

where:
*   **Q = {q₁, q₂, ..., qⱼ}** is a finite set of discrete modes representing the discrete states of H.
*   **V = X ⨂ U** is a finite set of continuous variables. It consists of:
    *   (observable) output variables **X = {x₁, x₂, ..., xₙ}**
    *   (controllable) input variables **U = {u₁, u₂, ..., uₘ}**
    *   A real-valued vector `v` ∈ ℝⁿ⁺ᵐ denotes a continuous state. The overall state space is Q × ℝⁿ⁺ᵐ.
*   **F = {Fq}q∈Q** assigns a flow function `Fq` to each mode `q`, characterizing the change of output variables X over inputs U.
*   **Inv ⊆ Q × ℝⁿ⁺ᵐ** specifies the mode invariants representing admissible states of H.
*   **Init ⊆ Inv** is the initial condition defining admissible initial states.
*   **T ⊆ Q × G × R × Q** is the set of transition relations between modes, where:
    *   **G** is a set of guards `g` ⊆ ℝⁿ⁺ᵐ.
    *   **R** is a set of resets (updates) `r`: ℝⁿ⁺ᵐ → ℝⁿ. A transition is triggered immediately when its guard is active.