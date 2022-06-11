Now, having sampled our data to classify, let us describe our classification task. We have been given data $(x_i,\, y_i)_{i=1}^n$, where $y_i \in \{-1,1\}$ for all $i$. and we wish to find a function $f$ such that $f(x_i) = y_i$, at least approximately -- and in particular we are going to try to accomplish this task by means of Support Vector Machines.

In the very simplest case, we naïvely assume there really exists such a function, and it can be written on the form $f(x) = \mathrm{sgn}(\langle x,x'\rangle + b)$ for some $x' \in \mathbb{R}^2$ and $b \in \mathbb{R}$. This is the linearly separable case of a Support Vector Machine, without any kernel. Now, the requirement that $f(x_i) = y_i$ then becomes $\mathrm{sgn}(\langle x_i, x'\rangle + b) = y_i$, or equivalently $\mathrm{sgn}(\langle x_i, x'\rangle + b)y_i > 0$. Of course, in the case where there exists such $x'$ and $b$, so the data genuinely is linearly separable, there will generically be infinitely many choices of the parameters -- so we need to make a choice, and the choice made by the SVM method is to maximise the distance between the hyperplane separating the two regions and the closest point in the training data.

If you carry through an analysis of what this means mathematically, you end up with the following optimization problem for linearly separable hard-margin SVM:

$$\min_{x', b} \frac{1}{2} \lVert x' \rVert^2, \qquad \text{ subject to: } y_i(\langle x_i, x'\rangle + b) \geq 1, \, \forall i \in [n].$$

Of course, as soon as we venture out of the domain of linearly separable data, we will find that there is *no* choice of $x'$ and $b$ satisfying that constraint -- so we need to move from hard-margin to soft-margin the moment we start introducing label noise or data that genuinely isn't generated from a linearly separable distribution.

The way to do this is by introducing *slack variables* $\xi_i$ which measure how much each constraint is violated, and a *cost* or *regularization* parameter $C$. Again skipping over a lot of analysis, the resulting optimization problem is now

$$
\min_{x', b} \frac{1}{2} \lVert x' \rVert^2 + C\sum_{i=1}^n \xi_i^2, \qquad \text{ subject to: } y_i(\langle x_i, x'\rangle + b) \geq 1 \text{ and } \xi_i \geq 0, \, \forall i \in [n].
$$

As usual in these kinds of problem, $C$ encodes the tradeoff between fitting training data perfectly and getting a model which generalises.

Now, so far, we have not changed our hypothesis class at all -- our prediction function will still just be the same old indicator of a halfspace. All we have done is to make the algorithm able to cope with a slight misspecification, still getting sensible results when there is some label noise or a slight curve to the separating line. Meanwhile, our actual data is not even approximately a halfspace -- so we need to do something radical.

So let us pick some cleverly chosen map $\Phi: \mathbb{R}^2 \to \mathbb{H}$ where $\mathbb{H}$ is an infinite dimensional Hilbert space, and instead of classifying by halfspaces in $\mathbb{R}^2$ we try to classify by halfspaces in $\mathbb{H}$. There is an obvious advantage to this -- infinity is considerably greater than two, so with the extra dimensions, there is plenty of room for what was previously not a linearly separable set to become linearly separable. So our new classifier will be $\mathrm{sgn}(\langle \Phi(x), x' \rangle + b)$ for some $x' \in \mathbb{H}$ and $b \in \mathbb{R}$.

There is also an obvious drawback to this -- infinity is actually *very very large*, and our computers don't know how to compute with infinite objects. Fortunately, there is a way around this problem, and that way proceeds via the unreasonable effectiveness of linear algebra in basically everything. In particular, it turns out that everything we need to know about the map $\Phi$ is actually encoded by the *kernel* $K_\Phi: \mathbb{R}^2 \times \mathbb{R} \to \mathbb{R}: (x, y) \mapsto \langle \Phi(x), \Phi(y)\rangle$, so if we specify this $K_\Phi$ in a way that we can compute, we need not worry about any infinities, or even exactly what $\Phi$ is at all -- at least as long as we made sure to choose $K_\Phi$ in such a way that there *could* be a $\Phi$ that generated it.

After proceeding through some beautiful functional analysis results, which I skip in the interest of brevity, we arrive at that we do not need to specify an $x' \in \mathbb{H}$, only an $\alpha \in \mathbb{R}^n$, and we get a classifier

$$
h(x) = \mathrm{sgn}\left(\sum_{i=1}^n \alpha_i y_i K_\Phi(x, x_i) + b\right).
$$

Further, our choice of $\alpha$ is given by the solution to the following eminently finite-dimensional optimization problem

$$
\max_\alpha \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j = 1}^n \alpha_i \alpha_j y_i y_j K_\Phi(x_i, x_j), \qquad \text{ subject to: } 0 \leq \alpha_i \leq C\quad \forall i\in [n],\quad \sum_{i=1}^n \alpha_i y_i = 0 
$$

and we get $b = y_i - \sum_{j=1}^n \alpha_j y_j K_\Phi(x_i, x_j)$ for any $i$ such that $0 < \alpha_i < C$.

In particular, there are four kernels implemented in this document:

1.  The trivial linear kernel $K(x, y) = \langle x, y \rangle$, giving us the linearly separable soft-margin case.

2.  The polynomial kernel $K(x,y) = (\gamma \langle x, y \rangle + k)^d$.

3.  The radial kernel $K(x,y) = e^{-\gamma\, \langle x, y\rangle^2}$.

4.  The sigmoid kernel $K(x,y) = \mathrm{tanh}(\gamma\langle x, y\rangle + k)$, which corresponds to training a very simple neural network on our data.

As the reader will have noticed, every non-trivial kernel here involves one or more hyperparameters like $\gamma$, $k$, and $d$. These will of course change what classifier we get, and so they can be tuned for best results, just like the regularization $C$.

Now, having given this brisk tour of the theory, let us actually play with it -- below, you can find a drop-down menu and some sliders to choose a kernel and the values of the hyperparameters, and a plot of what the resulting classifier is and how well it performs. In particular, you get to see both its performance on the training data and on some test data, to see the effect of things like regularization.
