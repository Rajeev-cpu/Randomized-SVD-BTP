\documentclass[aspectratio=169]{beamer}

% ---------------------------------------------------------
% Packages
% ---------------------------------------------------------
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}
\usepackage{booktabs}
\usepackage{tikz}
\usepackage{graphicx}
\usetikzlibrary{arrows.meta, positioning, shapes.geometric}

% ---------------------------------------------------------
% Theme
% ---------------------------------------------------------
\usetheme{Madrid}
\usecolortheme{default}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{footline}[frame number]
\setbeamertemplate{caption}[numbered]

% ---------------------------------------------------------
% Title information
% ---------------------------------------------------------
\title[Randomized SVD]{Randomized Singular Value Decomposition for\\
Efficient Low-Rank Matrix Approximation}
\subtitle{Theory, Implementation, Performance Comparison, and Image Compression}

\author[]{Student Name \\ \small Roll Number: XXXXXXXX}
\institute[]{
  Department of \emph{[Department Name]} \\
  \emph{[Institute Name]} \\[4pt]
  \small Supervisor: Dr.\ \emph{[Supervisor Name]}
}
\date{Mid-Semester Evaluation \\ \emph{[Date]}}

\begin{document}

% ===========================================================
% SLIDE 1 -- TITLE
% ===========================================================
\begin{frame}
  \titlepage
\end{frame}

% ===========================================================
% SLIDE 2 -- MOTIVATION AND PROBLEM STATEMENT
% ===========================================================
\begin{frame}{Motivation and Problem Statement}
  \textbf{The problem.} For a large matrix $A \in \mathbb{R}^{m\times n}$,
  computing the full classical SVD can become computationally expensive
  when only a small number of dominant singular components are actually
  required.

  \vspace{6pt}
  \textbf{Motivation}
  \begin{itemize}
    \item Image processing \quad $\bullet$ \ Data analysis \quad $\bullet$ \ Scientific computing
    \item Machine learning \quad $\bullet$ \ Numerical linear algebra
    \item These applications often produce matrices with an
          \textbf{approximately low-rank structure}.
    \item Often only a few dominant singular directions are important.
    \item Randomized SVD (RSVD) uses random projections to identify an
          approximate dominant subspace.
    \item This can reduce both computational work and storage when the target rank $k$ is
          small relative to $\min(m,n)$.
  \end{itemize}

  \vspace{6pt}
  \begin{block}{Project Objective}
    \small To study, implement, and experimentally evaluate Randomized SVD
    for efficient low-rank matrix approximation, focusing on approximation
    quality, oversampling, spectral decay, power iteration, computational
    cost, and image compression.
  \end{block}
\end{frame}

% ===========================================================
% SLIDE 3 -- BEST RANK-k APPROXIMATION AND TRUNCATED SVD
% ===========================================================
\begin{frame}{Best Rank-$k$ Approximation and Truncated SVD}
  \begin{columns}[T]
    \begin{column}{0.58\textwidth}
      \footnotesize
      \[
        \boxed{\min_{\operatorname{rank}(B)\leq k}\|A-B\|}
      \]
      \vspace{-4pt}
      The word ``best'' requires an error measure:
      \[
        \boxed{\min_{\operatorname{rank}(B)\leq k}\|A-B\|_2}
        \qquad
        \boxed{\min_{\operatorname{rank}(B)\leq k}\|A-B\|_F}
      \]

      \vspace{-4pt}
      From the SVD $A = U\Sigma V^T$:
      \[
        A=\sigma_1u_1v_1^T+\sigma_2u_2v_2^T+\cdots+\sigma_ru_rv_r^T
      \]
      \vspace{-4pt}
      Keeping the first $k$ terms:
      \[
        \boxed{A_k=U_k\Sigma_kV_k^T=\sum_{i=1}^{k}\sigma_i u_i v_i^T}
      \]

      \vspace{-4pt}
      \textbf{Eckart--Young--Mirsky:} the truncated SVD gives the best
      rank-$k$ approximation in both the spectral and Frobenius norms:
      \[
        \boxed{\|A-A_k\|_2=\sigma_{k+1}}
        \qquad
        \boxed{\|A-A_k\|_F=\sqrt{\textstyle\sum_{i=k+1}^{r}\sigma_i^2}}
      \]
    \end{column}
    \begin{column}{0.4\textwidth}
      \centering
      \begin{tikzpicture}[node distance=8mm, every node/.style={font=\small}]
        \node[draw, rounded corners, minimum width=2cm] (A) {$A$};
        \node[draw, rounded corners, minimum width=2cm, below=of A] (svd) {SVD};
        \node[draw, rounded corners, minimum width=2.6cm, below=of svd] (trunc) {retain first $k$};
        \node[draw, rounded corners, minimum width=2cm, below=of trunc] (Ak) {$A_k$};
        \draw[-{Latex}] (A) -- (svd);
        \draw[-{Latex}] (svd) -- (trunc);
        \draw[-{Latex}] (trunc) -- (Ak);
      \end{tikzpicture}
      \vspace{10pt}

      \begin{alertblock}{Key Point}
        \scriptsize The optimal truncated SVD is the benchmark against
        which the RSVD approximation will be evaluated.
      \end{alertblock}
    \end{column}
  \end{columns}
\end{frame}

% ===========================================================
% SLIDE 4 -- FROM A RANDOM VECTOR TO A\Omega
% ===========================================================
\begin{frame}{Why Does Multiplication by $A$ Reveal Dominant Directions?}
  \footnotesize
  Start with a random vector $\omega\in\mathbb{R}^n$, expressed in the
  right singular-vector basis:
  \[
    \omega=\sum_{i=1}^{r}c_i v_i, \qquad c_i\sim\mathcal N(0,1)\ \text{(Gaussian)}
  \]

  \vspace{-6pt}
  Apply $A$, and use $Av_i=\sigma_i u_i$:
  \[
    A\omega=A\sum_{i=1}^{r}c_i v_i
    \;\;\Longrightarrow\;\;
    \boxed{A\omega=\sum_{i=1}^{r}c_i\sigma_i u_i}
  \]

  \vspace{8pt}
  \begin{columns}[T]
    \begin{column}{0.5\textwidth}
      \textbf{Before multiplication:}
      \[
        \omega=c_1v_1+c_2v_2+\cdots+c_rv_r
      \]
    \end{column}
    \begin{column}{0.5\textwidth}
      \textbf{After multiplication:}
      \[
        A\omega=c_1\sigma_1u_1+c_2\sigma_2u_2+\cdots+c_r\sigma_ru_r
      \]
    \end{column}
  \end{columns}

  \vspace{2pt}
  \[
    \boxed{v_i \xrightarrow{\;A\;} \sigma_i u_i}
  \]

  \vspace{-4pt}
  \begin{itemize}
    \item The random vector has \emph{random coefficients} in the
          singular-vector basis --- it does not itself ``contain'' the
          singular vectors.
    \item Multiplication by $A$ maps each right singular direction $v_i$
          onto the corresponding left singular direction $u_i$, scaled by $\sigma_i$.
    \item Large singular values amplify their corresponding directions more strongly.
  \end{itemize}

  \vspace{-2pt}
  \begin{center}
    \boxed{A\omega \text{ tends to emphasize dominant left singular directions.}}
  \end{center}
\end{frame}

% ===========================================================
% SLIDE 5 -- RANDOMIZED RANGE FINDING
% ===========================================================
\begin{frame}{Randomized Range Finding}
  \small
  Generalize from one random vector to many: choose
  $\Omega\in\mathbb{R}^{n\times \ell}$ with
  \[
    \boxed{\ell=k+p}
  \]
  where $k$ = target rank, $p$ = oversampling, $\ell$ = dimension of the sampled subspace.

  \[
    \boxed{Y=A\Omega}
    \qquad\text{compute reduced QR:}\qquad
    Y=QR
  \]
  with $Q$ having orthonormal columns. Construct $P=QQ^T$ and approximate
  \[
    \boxed{A\approx QQ^TA}
  \]

  \begin{center}
    \begin{tikzpicture}[node distance=6mm, every node/.style={font=\scriptsize, draw, rounded corners}]
      \node (a) {$A+\Omega$};
      \node[right=of a] (y) {$Y=A\Omega$};
      \node[right=of y] (qr) {QR};
      \node[right=of qr] (q) {$Q$};
      \node[right=of q] (qqta) {$QQ^TA$};
      \draw[-{Latex}] (a) -- (y);
      \draw[-{Latex}] (y) -- (qr);
      \draw[-{Latex}] (qr) -- (q);
      \draw[-{Latex}] (q) -- (qqta);
    \end{tikzpicture}
  \end{center}

  \begin{alertblock}{Important Distinction}
    \scriptsize Because $\ell=k+p$, the matrix $QQ^TA$ can have rank up to
    $\ell$, \emph{not necessarily $k$}. $QQ^TA$ is a randomized subspace
    approximation; a final truncation step is required to obtain a strict
    rank-$k$ RSVD approximation.
  \end{alertblock}
\end{frame}

% ===========================================================
% SLIDE 6 -- WHY A\Omega CAPTURES THE DOMINANT SUBSPACE
% ===========================================================
\begin{frame}{Why $A\Omega$ Captures the Dominant Subspace}
  \small
  With $A=U\Sigma V^T$: $\quad Y=A\Omega=U\Sigma V^T\Omega$.

  Partition the SVD into dominant and trailing parts:
  \[
    A=U_1\Sigma_1V_1^T+U_2\Sigma_2V_2^T
  \]
  Define $\Omega_1=V_1^T\Omega,\ \Omega_2=V_2^T\Omega$. Then
  \[
    \boxed{Y=U_1\Sigma_1\Omega_1+U_2\Sigma_2\Omega_2}
  \]

  \begin{itemize}
    \item $U_1\Sigma_1\Omega_1$ is the dominant contribution; $U_2\Sigma_2\Omega_2$ is the trailing contribution.
    \item Large singular values in $\Sigma_1$ amplify the dominant part.
    \item If $\sigma_k \gg \sigma_{k+1}$, the dominant subspace is easier to capture.
  \end{itemize}

  \begin{center}
    \begin{tikzpicture}[node distance=6mm, every node/.style={font=\scriptsize, draw, rounded corners}]
      \node (o) {Random $\Omega$};
      \node[right=of o] (vo) {$V^T\Omega$};
      \node[right=of vo] (s) {$\Sigma$ scales components};
      \node[right=of s] (uv) {$U\Sigma V^T\Omega$};
      \node[right=of uv] (y) {$Y$};
      \draw[-{Latex}] (o) -- (vo);
      \draw[-{Latex}] (vo) -- (s);
      \draw[-{Latex}] (s) -- (uv);
      \draw[-{Latex}] (uv) -- (y);
    \end{tikzpicture}
  \end{center}
\end{frame}

% ===========================================================
% SLIDE 7 -- COMPLETE RSVD ALGORITHM
% ===========================================================
\begin{frame}{Complete RSVD Algorithm}
  \begin{columns}[T]
    \begin{column}{0.56\textwidth}
      \scriptsize
      \begin{enumerate}
        \item Generate Gaussian random matrix $\Omega\in\mathbb{R}^{n\times\ell}$
        \item $Y=A\Omega$
        \item $Y=QR$
        \item \textbf{Stable power iteration} (repeat $q$ times):
              \[
                Z=A^TQ,\ \ Z=Q_ZR_Z,\ \ Y=AQ_Z,\ \ \text{re-QR}\to Q
              \]
        \item $B=Q^TA$, \quad $B\in\mathbb{R}^{\ell\times n}$
        \item $B=\widetilde U\Sigma V^T$ \ (small SVD)
        \item $U=Q\widetilde U$
        \item Strict rank-$k$ truncation:
              \[
                \boxed{A_k^{RSVD}=U_k\Sigma_kV_k^T}
              \]
      \end{enumerate}
      \vspace{4pt}
      \textit{QR reorthogonalization is used during power iteration for numerical stability.}
    \end{column}
    \begin{column}{0.42\textwidth}
      \centering
      \begin{tikzpicture}[node distance=5mm, every node/.style={font=\scriptsize, draw, rounded corners, align=center}]
        \node (om) {$\Omega$};
        \node[below=of om] (y) {$Y$};
        \node[below=of y] (q) {$Q$};
        \node[below=of q] (pi) {Power\\Iteration};
        \node[below=of pi] (b) {$B$};
        \node[below=of b] (svd) {small SVD};
        \node[below=of svd] (u) {$U$};
        \node[below=of u] (rk) {rank-$k$ RSVD};
        \draw[-{Latex}] (om)--(y);
        \draw[-{Latex}] (y)--(q);
        \draw[-{Latex}] (q)--(pi);
        \draw[-{Latex}] (pi)--(b);
        \draw[-{Latex}] (b)--(svd);
        \draw[-{Latex}] (svd)--(u);
        \draw[-{Latex}] (u)--(rk);
      \end{tikzpicture}
    \end{column}
  \end{columns}
\end{frame}

% ===========================================================
% SLIDE 8 -- COMPUTATIONAL COMPLEXITY
% ===========================================================
\begin{frame}{Computational Complexity --- HMT-Based View}
  \scriptsize
  \textit{N. Halko, P.-G. Martinsson, J. A. Tropp, ``Finding Structure with
  Randomness: Probabilistic Algorithms for Constructing Approximate Matrix
  Decompositions,'' SIAM Review, 53(2), 217--288, 2011.}

  \vspace{4pt}
  \normalsize
  HMT basic randomized range finder cost model:
  \[
    \boxed{T_{\mathrm{basic}}\sim \ell n T_{\mathrm{rand}} + \ell T_{\mathrm{mult}} + \ell^2 m}
  \]
  \scriptsize
  $\ell n T_{\mathrm{rand}}$: random test matrix generation \quad
  $\ell T_{\mathrm{mult}}$: matrix application \quad
  $\ell^2 m$: orthonormalization term

  \normalsize
  \vspace{4pt}
  Dense operation-level interpretation:
  \[
    A\Omega:\ O(mn\ell) \qquad QR:\ O(m\ell^2) \qquad Q^TA:\ O(mn\ell)
  \]
  \[
    \boxed{T_{RSVD}\approx O(2mn\ell+m\ell^2)+T_{SVD}(\ell\times n)}, \qquad \ell=k+p
  \]

  Power iteration: $Y=(AA^T)^qA\Omega$ requires $2q+1$ applications of $A$ or $A^T$:
  \[
    \boxed{O((2q+1)mn\ell)} \ \text{(plus QR/orthogonalization and reduced-SVD costs)}
  \]

  \scriptsize
  \textit{The HMT expression is a computational cost model; the dense
  $O(\cdot)$ expressions above are operation-level specializations for this
  implementation, not literal HMT flop counts. Computational savings are
  associated with $\ell=k+p\ll\min(m,n)$; actual runtime depends on
  implementation and hardware.}
\end{frame}

% ===========================================================
% SLIDE 9 -- EXPERIMENTAL SETUP
% ===========================================================
\begin{frame}{Experimental Setup}
  \small
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \textbf{Environment}
      \begin{itemize}
        \item Python, NumPy, Matplotlib, VS Code
      \end{itemize}
      \textbf{Baseline target rank:} $k=20$

      \textbf{Oversampling:} $p=0,5,10,20,30$

      \textbf{Power iterations:} $q=0,1,2,3$

      \textbf{Matrix types}
      \begin{itemize}
        \item Gaussian random matrices
        \item Controlled matrices with prescribed singular-value spectra
      \end{itemize}
    \end{column}
    \begin{column}{0.5\textwidth}
      \textbf{Prescribed spectra}
      \[
        \text{Exponential: } \sigma_i=e^{-0.15(i-1)}
      \]
      \[
        \text{Polynomial: } \sigma_i=i^{-1}
      \]
      \[
        \text{Slow: } \sigma_i=i^{-0.1}
      \]
      \textbf{Metrics}
      \begin{itemize}
        \item Frobenius error, relative error
        \item Optimal rank-$k$ error, orthogonality error
        \item Runtime
        \item \textit{Future:} PSNR, compression ratio
      \end{itemize}
    \end{column}
  \end{columns}
\end{frame}

% ===========================================================
% SLIDE 10 -- EFFECT OF OVERSAMPLING
% ===========================================================
\begin{frame}{Effect of Oversampling}
  \small
  \begin{columns}[T]
    \begin{column}{0.52\textwidth}
      \centering
      \begin{tabular}{cccc}
        \toprule
        $p$ & $\ell$ & Frobenius Error & Relative Error \\
        \midrule
        0  & 20 & 367.202306 & 0.947678 \\
        5  & 25 & 361.350754 & 0.932576 \\
        10 & 30 & 356.744904 & 0.920689 \\
        20 & 40 & 346.204019 & 0.893485 \\
        30 & 50 & 335.445329 & 0.865719 \\
        \bottomrule
      \end{tabular}
    \end{column}
    \begin{column}{0.46\textwidth}
      \centering
      \begin{tikzpicture}[scale=0.85]
        \begin{scope}
          \draw[-{Latex}] (0,0) -- (5.2,0) node[right, font=\scriptsize] {$p$};
          \draw[-{Latex}] (0,0) -- (0,4.2) node[above, font=\scriptsize] {Rel. Error};
          % data points (p, scaled relative error)
          \foreach \x/\y in {0/3.79,1.25/3.73,2.5/3.68,5/3.57,7.5/3.46}{
            \fill (\x*0.6+0.3,\y) circle (1.5pt);
          }
          \draw[thick] (0.3,3.79) -- (1.05,3.73) -- (1.8,3.68) -- (3.3,3.57) -- (4.8,3.46);
          \node[font=\scriptsize] at (0.3,-0.35) {0};
          \node[font=\scriptsize] at (1.05,-0.35) {5};
          \node[font=\scriptsize] at (1.8,-0.35) {10};
          \node[font=\scriptsize] at (3.3,-0.35) {20};
          \node[font=\scriptsize] at (4.8,-0.35) {30};
        \end{scope}
      \end{tikzpicture}
      \scriptsize (Relative Projection Error vs.\ $p$)
    \end{column}
  \end{columns}

  \vspace{6pt}
  \small
  \textbf{Observation:} In this experiment, increasing the oversampling
  parameter $p$ reduced the projection error. The orthogonality error of
  $Q$ remained approximately at machine precision.

  \textit{This experiment uses a generic Gaussian full-rank matrix, so the
  relative approximation error is expected to be relatively high.}

  \vspace{4pt}
  \scriptsize\textit{Experiment code: oversampling\_experiment.py}
\end{frame}

% ===========================================================
% SLIDE 11 -- SINGULAR-VALUE DECAY AND POWER ITERATION
% ===========================================================
\begin{frame}{Singular-Value Decay and Power Iteration}
  \footnotesize
  \textbf{A. Singular-Value Decay} \quad
  Exponential: $\sigma_i=e^{-0.15(i-1)}$ \quad
  Polynomial: $\sigma_i=i^{-1}$ \quad
  Slow: $\sigma_i=i^{-0.1}$

  \centering
  \begin{tabular}{lcc}
    \toprule
    Spectrum & RSVD/Projection Rel.\ Error & Optimal Rank-20 Rel.\ Error \\
    \midrule
    Exponential & 0.03441 & 0.04979 \\
    Polynomial  & 0.20939 & 0.16638 \\
    Slow        & 0.94618 & 0.94272 \\
    \bottomrule
  \end{tabular}

  \raggedright
  \textbf{Observation:} RSVD performance strongly depends on the
  singular-value spectrum. Fast decay $\rightarrow$ stronger low-rank
  structure; polynomial decay $\rightarrow$ moderate difficulty; slow decay
  $\rightarrow$ weak low-rank structure.

  \textit{The current $QQ^TA$ experiment uses $\ell=k+p$, so its rank can
  exceed $k$; these are preliminary projection results, not a strict
  rank-$k$ RSVD-vs-optimal comparison. A strict rank-$k$ comparison will
  be performed using the complete RSVD and final truncation.}

  \scriptsize\textit{Experiment code: spectral\_decay\_experiment.py}

  \vspace{4pt}
  \footnotesize
  \textbf{B. Power Iteration} \quad
  $Y=(AA^T)^qA\Omega$, and with $A=U\Sigma V^T$: $(AA^T)^qA=U\Sigma^{2q+1}V^T$, so
  $Y=U\Sigma^{2q+1}V^T\Omega$. Amplification factor:
  $\boxed{\left(\sigma_k/\sigma_{k+1}\right)^{2q+1}}$

  \textit{Theoretical example} (if $\sigma_k/\sigma_{k+1}=2$): $q{=}0\to2$,
  $q{=}1\to8$, $q{=}2\to32$, $q{=}3\to128$ \quad (not experimental data).

  Trade-off: higher $q$ improves spectral separation, but adds matrix
  multiplications, runtime, and requires numerical stabilization.
\end{frame}

% ===========================================================
% SLIDE 12 -- LIMITATIONS, PROPOSED IMPROVEMENT, FUTURE WORK, CONCLUSION
% ===========================================================
\begin{frame}{Limitations, Proposed Improvement, Future Work, Conclusion}
  \scriptsize
  \begin{columns}[T]
    \begin{column}{0.5\textwidth}
      \textbf{Limitations}
      \begin{itemize}
        \item $k$ is currently manually selected
        \item $p$ affects accuracy and computational cost
        \item $q$ improves spectral separation but increases computation
        \item Performance depends on singular-value decay
        \item Randomized methods show trial-to-trial variation
      \end{itemize}

      \textbf{Proposed / Future Experiment}
      \begin{itemize}
        \item Compare fixed $k=20$ vs.\ automatically selected $k$
              (energy-, spectrum-, or error-based criterion)
        \item Vary $p=0,5,10,20,30$ and $q=0,1,2,3$
        \item Evaluate relative error, runtime, memory, approximation quality
      \end{itemize}
    \end{column}
    \begin{column}{0.5\textwidth}
      \textbf{Future Work}
      \begin{itemize}
        \item Complete strict rank-$k$ RSVD comparison
        \item Runtime / scalability experiments
        \item Image compression
        \item PSNR analysis
        \item Compression-ratio analysis
        \item Adaptive parameter selection
      \end{itemize}

      \textbf{Conclusion}
      \begin{quote}
        \scriptsize RSVD constructs a low-dimensional randomized subspace
        that can capture dominant matrix structure. Preliminary
        experiments demonstrate the importance of oversampling and
        singular-value decay, while power iteration provides a mechanism
        for improving subspace separation.
      \end{quote}
    \end{column}
  \end{columns}

  \vspace{8pt}
  \centering
  \Large \textbf{THANK YOU}
\end{frame}

\end{document}