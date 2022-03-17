from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
from matplotlib import pyplot as plt


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma = 10, 1
    samples = np.random.normal(mu, sigma, 1000)
    uvg_est = UnivariateGaussian()
    uvg_est.fit(samples)
    print(uvg_est.mu_, uvg_est.var_)

    # Question 2 - Empirically showing sample mean is consistent
    expectation_distances = np.zeros(100)
    for i in range(100):
        inc_uvg_est = UnivariateGaussian()
        inc_uvg_est.fit(samples[0:(i+1)*10])
        expectation_distances[i] = np.abs(inc_uvg_est.mu_ - mu)
    sample_size = np.linspace(10, 1000, 100)
    plt.figure()
    plt.plot(sample_size, expectation_distances)
    plt.title("Sample-based estimation of expectation")
    plt.xlabel("Sample size")
    plt.ylabel("Abs dist from actual expectation")
    plt.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf = uvg_est.pdf(samples)
    plt.figure()
    plt.scatter(samples, pdf)
    plt.title("Scatter PDF of 1000 samples from N(10, 1) distribution")
    plt.xlabel("Sample value")
    plt.ylabel("PDF ('Probability') of sample value")
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0], dtype=np.float64)
    cov = np.array([[1  , 0.2, 0, 0.5],
                    [0.2, 2  , 0, 0  ],
                    [0  , 0  , 1, 0  ],
                    [0.5, 0  , 0, 1  ]], dtype=np.float64)
    samples = np.random.multivariate_normal(mu, cov, 1000)
    mvg_est = MultivariateGaussian()
    mvg_est.fit(samples)
    print(mvg_est.mu_)
    print(mvg_est.cov_)

    # Question 5 - Likelihood evaluation
    heatmap = np.zeros((200, 200))
    expectation_vals = np.linspace(-10, 10, 200)
    for i in range(200):
        for j in range(200):
            mu[0] = expectation_vals[j]
            mu[2] = expectation_vals[i]
            heatmap[i][j] = mvg_est.log_likelihood(mu, cov, samples)
    plt.figure()
    plt.imshow(np.flip(heatmap, axis=0), cmap='plasma', extent=[-10, 10, -10, 10])
    plt.colorbar()
    plt.title("Heatmap of likelihood of f1 and f3 expectation values")
    plt.xlabel("Likelihood of f1")
    plt.ylabel("Likelihood of f3")
    plt.show()

    # Question 6 - Maximum likelihood
    f3_ind, f1_ind = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    f1 = expectation_vals[f1_ind]
    f3 = expectation_vals[f3_ind]
    max_ll = heatmap[f3_ind][f1_ind]
    print("argmax for f1: ", f1)
    print("argmax for f3: ", f3)
    print("maximal log-likelihood is: ", max_ll)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
