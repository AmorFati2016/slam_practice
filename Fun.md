# The normalized 8-point algorithm for F.
Objective\
Given n ≥ 8 image point correspondences {$x_i$ ↔ $x'_i$}, determine the fundamental matrix F
such that $x'^T_iFx_i$ = 0.
Algorithm\
Step1. Normalization: Transform the image coordinates according to $x̂_i = Tx_i$ and $x̂'_i$ =
$Tx'_i$ , where T and T' are normalizing transformations consisting of a translation and
scaling.

Step2. Find the fundamental matrix F̂ corresponding to the matches $x̂_i$ ↔ $x̂'_i$ by
(a) Linear solution: Determine F̂ from the singular vector corresponding to the
smallest singular value of Â, where Â is composed from the matches x̂ i ↔ x̂  i
as defined in (11.3).
(b) Constraint enforcement: Replace F̂ by F̂ such that det F̂ = 0 using the SVD
(see section 11.1.1).
Step3. Denormalization: Set F = $T'^TF̂T$. Matrix F is the fundamental matrix corresponding
to the original data x i ↔ x  i .