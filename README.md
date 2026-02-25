# Online-to-Batch Conversion: Empirical Demonstration

In classical machine learning, **Empirical Risk Minimization (ERM)** is the dominant paradigm, where a model is trained by minimizing the average loss over a fixed batch of i.i.d. data. While ERM offers strong theoretical guarantees, it requires solving a potentially complex optimization problem over the entire dataset at once.

**Online Learning** offers an alternative perspective. Data arrives sequentially and the model updates incrementally to minimize **regret**. However, online learning outputs a sequence of predictors rather than a single solution, in convex scenarios they are averaged.

The **Online-to-Batch (O2B)** framework serves as a theoretical bridge between these two worlds. As discussed in *Orabona (2025)*, O2B conversion demonstrates that:
1. An online algorithm with sublinear regret can be converted into a batch predictor with strong generalization guarantees.
2. Online learning matches the sample complexity of ERM under i.i.d. assumptions.
3. It provides a valid stochastic optimizer that avoids the need for exact empirical risk minimization.

This repository implements a practical demonstration of this theory, showing how an online learning algorithm, when combined with uniform averaging, achieves performance comparable to batch ERM despite processing data sequentially.

## Experiment Setup

This implementation reproduces the theoretical setting described in **Theorem 3.1** (*Orabona, 2025*) focusing on convex loss functions.

### 1. Problem Formulation
*   **Task:** Binary Linear Classification.
*   **Hypothesis Space:** $\mathcal{V} \subseteq \mathbb{R}^d$ (Convex set).
*   **Loss Function:** **Logistic Loss** (Convex, differentiable), defined as:
    $$ \ell(x; (z, y)) = \ln(1 + \exp(-y \langle x, z \rangle)) $$
    *(Refer to Example 3.4 in `online_learning_to_batch_learning.pdf`)*.
*   **Data Assumption:** Samples $(z_t, y_t)$ are drawn **i.i.d.** from an underlying distribution $\rho$ *(Assumption 2.1, Seminar Report)*.

### 2. Algorithms Compared
1.  **Online Gradient Descent (OGD):**
    *   Processes samples one-by-one.
    *   Updates weights using sequential gradients with learning rate decay $\eta_t \propto 1/\sqrt{t}$.
    *   Guarantees sublinear regret $O(\sqrt{T})$.
2.  **Online-to-Batch Converter (Averaging):**
    *   Constructs a single predictor $\bar{x}_T$ by uniformly averaging the sequence of online iterates:
        $$ \bar{x}_T = \frac{1}{T} \sum_{t=1}^T x_t $$
    *   Relies on **Jensen's Inequality** due to the convexity of the loss *(Theorem 3.1)*.
3.  **Empirical Risk Minimization (ERM):**
    *   **Batch Baseline:** Trains via batch optimization (e.g., LBFGS) on the full training set.
    *   Serves as the benchmark for optimal empirical risk.

### 3. Evaluation Metrics
To validate the theoretical bounds, the experiment tracks three key metrics over increasing sample sizes $T$:

*   **Cumulative Regret:** Verifies the online algorithm achieves sublinear growth ($Regret_T = o(T)$), which is the prerequisite for conversion.
*   **Empirical Risk (Training Error):** Compares the risk of the averaged online predictor against the ERM solution. Theory predicts the gap vanishes as $O(1/\sqrt{T})$.
*   **True Risk (Test Error):** Evaluates generalization on a test set. This demonstrates that the online-to-batch predictor generalizes as well as the batch ERM solution, validating the **PAC-learning** connection *(Section 3.2, Orabona 2025)*.

## Expected Results

Running the simulation should produce plots illustrating the following theoretical claims:
1.  **Regret Plot:** The cumulative regret curve should grow sublinearly (below the $O(\sqrt{T})$ reference line), confirming the "no-regret" property.
2.  **Risk Convergence:** The training risk of the **Averaged Online** predictor should converge towards the **ERM** risk as $T$ increases. The **Last Iterate** of the online algorithm may oscillate, highlighting the necessity of averaging for convex conversion.
3.  **Generalization:** The test error of the Online-to-Batch predictor should closely track the ERM test error, demonstrating that sequential learning does not sacrifice statistical efficiency.

## References
*   **Orabona, F. (2025).** *A Modern Introduction to Online Learning.* Chapter 3: Online-to-Batch Conversion. (Specifically Theorem 3.1 and Example 3.4).

## How to Run

1.  Ensure Python 3.x is installed with standard scientific libraries (`numpy`, `matplotlib`, `scikit-learn`).
2.  Run the Jupyter notebook to reproduce all results or execute the main script:
    ```bash
    python online_to_batch_demo.py
    ```
3.  Observe the generated plots comparing Training Risk, Test Risk, and Regret.