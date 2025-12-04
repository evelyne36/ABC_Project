# Generalized gradient boosting library
## Review of xgboost
the Taylor expansion of the loss function at t step is:
$$\sum_{i=1} ^n [g_if_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)] + \Omega(f_t)$$.
In most cases, we assume f_t(x_i) is a constant for both left and right trees,
let w_j = f_t(x_i), so we have
\begin{equation}
L = & \sum_{i=1} ^n [g_if_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)] + \Omega(f_t) \\
  = & \sum_{i=1} ^n [(\sum_{i \in I_j} g_i) w_j + \frac{1}{2}(\sum_{i \in I_j} w_j^2)] + \Omega(f_t) \\    
\end{equation}
For a fixed structure, we can compute the optimal weight w_j^* of leaf j by:
$$
w_j^* = -\frac{\sum_{i \in I_j} g_i} {\sum_{i\in I_j}h_i + \lambda}
$$
In regression, h_i = 1 for non weight cases, g_i = y - \hat{y}^{t-1}, which is simply the residual
of the y, so, the optimal weight for a given leaf is the mean of the residual of the y.

## Conditional variable vs Predictive variable
We can treat the original split of the tree node as using conditional variable and predictive variables.
We use conditional variables to decide on the split, and use the predictive variable to evaluate the 
the split. In the original gradient boosting, the predictive variable is simply a bias, i.e., w_j for a 
given leaf j.

In some cases, the conditional variables has no prediction power at all, using a bias as predictive variables
seems too weak. 

Consider the classical example below:
y axis
^
| x o
| o x
--------> x axis
We want to classifiy x from o, neither the two axis, x and y, is informative. The loss will not reduce if we use
either x or y as decision variables. so we will end up with a two level trees:
   x <1
  /  \
y<1  y< 1
/ \  /  \
o  x x  o

However, suppose we are able to use variables as predictive variables, we only need to build a one-level tree
to classify the examples:
   x <1
  /  \
y -1 1-y
This make the decision much more efficient!

## Generalized gradient boosting

f_t(x_i) is a bias in xgboost, this gives us significant speed up in learing the tree algorithm. 
We want to generalize it so that f_t(x_i) = \beta x_ij, (x_ij is feature j of sample x_i). Actually
f_t(x_i) can be any functions, but we want to maintain the speed vs expressiveness trade off, more
complex function will decrease the speed and be more likely to overfit.

With this, we get: 
\begin{equation}
L = & \sum_{i=1} ^n [g_i \beta x_ij + \frac{1}{2}h_i  \beta^2 x_ij^2] + \Omega(f_t) \\
  = & \sum_{i=1} ^n [(\sum_{i \in I_j} g_i) w_j + \frac{1}{2}(\sum_{i \in I_j} w_j^2)] + \Omega(f_t) \\    
\end{equation}
For a fixed structure, we can compute the optimal weight \beta of leaf j as
$$
\beta^* = -\frac{\sum_{i \in I_j} g_i x_ij} {\sum_{i\in I_j}h_i x_ij^2 + \lambda}
$$

For every dataset, we add a bias into the feature and use the bias as the predictive variable in every step of splitting,
then we fall back to the original algorithm.


