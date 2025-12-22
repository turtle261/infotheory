# Information Dimensionality

1. NCD
NCD_vitanyi(x,y) =[(C(xy) - min(C(x), C(y))) / max(C(x), C(y))]
NCD_sym_vitanyi(x,y) = [ ( min(C(xy), C(yx)) - min(C(x), C(y)) ) / max(C(x), C(y)) ]
NCD_sym_cons(x,y) = [ ( min(C(xy), C(yx)) - min(C(x), C(y)) ) / min(C(xy), C(yx)) ]
NCD_cons(x,y) = [ ( C(xy) - min(C(x), C(y)) ) / C(xy) ]


2. NED  - Normalized Entropy Distance

# NED measures the fraction of the larger variable’s uncertainty that remains after observing the smaller one.
# If you need to know “How much of the information in the more complex variable is not explained by the simpler one?”

Define: [
p(x) = Pr(X = x) with sum_x p(x) = 1
p(y) = Pr(Y = y) with sum_y p(y) = 1
p(x,y) = Pr(X = x, Y = y) with sum_{x,y} p(x,y) = 1
H(X) = - sum_x p(x) * log p(x)
H(Y) = - sum_y p(y) * log p(y)
H(X,Y) = - sum_{x,y} p(x,y) * log p(x,y)

]

NED(X,Y) = [ ( H(X,Y) - min(H(X), H(Y)) ) / max(H(X), H(Y)) ] # Should be equivalent to NED_MI
NED_cons(X,Y) = [ ( H(X,Y) - min(H(X), H(Y)) ) / H(X,Y) ]
NED_MI(X,Y) = [ 1 - I(X;Y) / max(H(X), H(Y)) ] where [I(X;Y) = H(X) + H(Y) - H(X,Y)] # Should be equivalent to NED

3. Normalized Transform Effort (effort to make x into y)
NTE(X,Y) = [VI(X,Y) / max(H(X), H(Y)) ] # VI = Variation of Information = Metric Quality.
where [VI(X,Y) = H(X|Y) + H(Y|X)]
where [H(X|Y) = H(X,Y) - H(Y)]
where [H(Y|X) = H(X,Y) - H(X)]

4. Total Variation Distance (L1-style probabilistic overlap)

Define: [
# For marginal distributions over the same support
p_X(i) = Pr(X = i) with sum_i p_X(i) = 1
p_Y(i) = Pr(Y = i) with sum_i p_Y(i) = 1
]

TVD_marg(X,Y) = [ (1/2) * sum_i | p_X(i) - p_Y(i) | ]  # Total Variation Distance (true metric, [0,1])
# Interpretation: maximum difference in probability assigned to any event; measures worst-case discriminability.
5. Hellinger-based Distances

Define: [
# For marginal distributions (treating X and Y as separate PMFs over the same support)
p_X(i) = Pr(X = i) with sum_i p_X(i) = 1
p_Y(i) = Pr(Y = i) with sum_i p_Y(i) = 1

BC(X,Y) = sum_i sqrt( p_X(i) * p_Y(i) )  # Bhattacharyya Coefficient (similarity, range [0,1])
]

NHD_marg(X,Y) = [ sqrt( 1 - BC(X,Y) ) ]  # Normalized Hellinger Distance (equivalent to standard Hellinger / sqrt(2) scaled to [0,1], true metric)
# Or equivalently:
NHD_marg(X,Y) = [ (1 / sqrt(2)) * sqrt( sum_i ( sqrt(p_X(i)) - sqrt(p_Y(i)) )^2 ) ]

# Alternative common form (squared Hellinger, bounded [0,1]):
HD_sq_marg(X,Y) = [ (1/2) * sum_i ( sqrt(p_X(i)) - sqrt(p_Y(i)) )^2 ]  # = 1 - BC(X,Y)


6. Intrinsic vs Extrinsic Dependence (how compressed the data already is relative to it's priors)
- This measures(in practice approximates) how compressed the data already is. If it is heavily extrinsically dependent, it is compressed by it's priors, and if it is highly intrinsically dependent, the opposite is true (e.g. Periodicity/Symmetry)


7. Similarity/resistance under allowed transformations (How much noise until x stops being x)
- Resistance to transformation. Transformation MAY be noise, or may be explicitly defined rules (for example swapping words with synonyms in Natural Language data)
