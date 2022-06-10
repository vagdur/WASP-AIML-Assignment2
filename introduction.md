In this assignment, we are going to explore how Support Vector Machines (SVMs) learn non-linear data. In fact, because of the much greater possibilities of visualising things in an interactive format like this, compared to a static pdf, we will go a bit further than the assignment asked, in the spirit of curiosity.

So, first off, let us generate some data! Our data will be drawn from a $\mathcal{N}\left(\begin{pmatrix}0 \\ 0\end{pmatrix},\begin{pmatrix}1 & \rho \\ \rho & 1 \end{pmatrix}\right)$ distribution -- that is, a bivariate normal distribution with mean zero, unit variances, and correlation $\rho$ between the two coordinates. The reason for introducing a correlation between the two coordinates is that this will bias our distribution of true-false for our classification problem, so we can see how the SVM deals with having only few examples of one class.

Second, we will also allow some *label noise* -- that is, we introduce a parameter $\epsilon$, and say that each data point gets mislabeled with probability $\epsilon$. This will let us see how the algorithm copes with this kind of data error. Of course none of the kernels we will consider actually have the ground truth in its hypothesis space, so even without data error we will expect an accuracy below $100\%$, but it is still interesting to see how much lower accuracy we get.

Finally, we also add a slider for $n$, the number of data points -- now to explore how the algorithms cope with very low amounts of data, or conversely to see how much better they do when given much more data.

All three parameters are tunable via the sliders below, and their default values are set to what the assignment asked for, $n = 200$ and $\epsilon = \rho = 0$.
