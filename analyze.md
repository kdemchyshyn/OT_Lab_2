## **Discussion**

Does the number of iterations in both methods depend on the starting point for the problem
under consideration?

Answer: Yes, in both methods thr number of iterations is different based on starting point. But Newton method can be independent in case of positively defined quadratic function.

How many iterations are needed for Newton’s method to converge for a positive definite
quadratic form using exact second derivative information?

Answer: 1 iteration

Why does the BFGS update not converge to the true Hessian?

Answer: BFGS is just an approximation for true inverse Hessian. It does not uses all information about the function. That's why it does not converge to the true Hessian, unless this is quadratic function.

Is the BFGS approximation always positive definite?

Answer: Yes, it is.

Compare the number of iterations spent to find the solution in both methods.

Answer: 
Newton method is faster then Quasi Newton, but it is more costly in terms of computation resources.

| x0_point | Newton | Quasi Newton |
|----------|--------|--------------|
| [2, 4]   | 2      | 1076         |
| [-2, 10] | 5      | 6            |