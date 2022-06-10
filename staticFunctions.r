# This file contains functions that do the various things needed in the assignment. Then they are called from app.r, which does
# all the nice interactive stuff.

# This function generates the data which we will use to train our SVM. By default we generate 200 points from a standard
# normal distribution on R^2, but we can also make correlated points and change the number of points:
generateData <- function(n = 200, rho = 0) {
  # There is of course a library function to do exactly this:
  return(MASS::mvrnorm(n = n,
                       mu = c(0,0),
                       Sigma = matrix(c(1,rho,rho,1), nrow = 2)))
} 

# This function computes labels for the data. By default there is no noise, and it just returns a vector
# of if points are in the first/third quadrant or not, but it can be tuned to have some label noise.
computeDataLabels <- function(x, epsilon = 0) {
  # First, we sample the random noise:
  randomNoise <- sample(c(-1,1),
                        size = dim(x)[1],
                        replace = TRUE,
                        prob = c(epsilon, 1 - epsilon))
  # Then, we can compute whether the two coordinates are in the same quadrant or not by just multiplying
  # them together and checking if it is positive or not - and we can add in our random noise by multiplying
  # it in as well:
  return(x[,1]*x[,2]*randomNoise > 0)
} 
