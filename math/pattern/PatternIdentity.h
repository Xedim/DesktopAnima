#pragma once

enum class PatternIdentity {
    // ---------------- Algebra / Arithmetic ----------------
    Factorial,
    Binomial,
    Combination,
    Permutation,
    Mod,
    GeometricSum,
    Polynomial,
    Rational,
    AlgebraicRoot,
    Sqrt,
    Cbrt,

    // ---------------- Power / Exponential ----------------
    Pow,
    Sign,
    Abs,
    Exp,
    Exp2,
    Expm1Safe,

    // ---------------- Logarithmic ----------------
    Log2,
    Log,
    Log10,
    LogA,
    Log1p,

    // ---------------- Trigonometric ----------------
    Sin,
    Cos,
    Sec,
    Csc,
    Sinc,
    Tan,
    Cot,
    Asin,
    Acos,
    Atan,
    Atan2,
    Hypot,

    // ---------------- Hyperbolic ----------------
    Sinh,
    Cosh,
    Sech,
    Csch,
    Tanh,
    Coth,
    Asinh,
    Acosh,
    Atanh,

    // ---------------- Hybrid / Numerical ----------------
    XPowY,
    Sqrt1pM1,
    Heaviside,

    // ---------------- Special Functions ----------------
    Erf,
    Erfc,
    Gamma,
    LGamma,
    Beta,
    CylBesselJ,
    CylNeumann,
    CylBesselI,
    CylBesselK,
    LambertW,
    Legendre,
    AssocLegendre,
    RiemannZeta,
    Zeta,

    // ---------------- Generalized ----------------
    DiracDelta,

    // ---------------- Numerical / Misc ----------------
    Round,
    Floor,
    Ceil,
    Trunc,
    Clamp,
    Lerp,
    Fma,

    // ---------------- Dynamical Systems ----------------
    TakensMap,

    // ---------------- Fractals ----------------
    Weierstrass,
    Cantor,
    Logistic,
    Tent,
    Julia,
    Escapes,
    Iterate,

    // ---------------- Descriptive Statistics ----------------
    Sum,
    Mean,
    Median,
    Mode,
    Min,
    Max,
    Range,
    Variance,
    VarianceUnbiased,
    StdDev,
    StdDevUnbiased,
    MeanAbsoluteDeviation,

    // ---------------- Shape Statistics ----------------
    Skewness,
    Kurtosis,
    Moment,
    RawMoment,

    // ---------------- Order & Quantiles ----------------
    Quantile,
    Percentile,
    Quartiles,
    IQR,
    TrimmedMean,

    // ---------------- Robust Statistics ----------------
    MedianAbsoluteDeviation,
    WinsorizedMean,
    HuberMean,
    BiweightMean,
    SNR,

    // ---------------- Correlation & Dependence ----------------
    Covariance,
    CorrelationPearson,
    CorrelationSpearman,
    CorrelationKendall,
    Autocorrelation,
    CrossCorrelation,

    // ---------------- Probability Distributions ----------------
    DistNormalPDF,
    DistNormalCDF,
    DistNormalQuantile,
    DistNormalLogLikelihood,

    // ---------------- Statistical Tests ----------------
    ZTest,
    TTest,
    WelchTTest,
    MannWhitneyU,
    WilcoxonSignedRank,
    KSTest,
    ChiSquareTest,
    AndersonDarling,

    // ---------------- Entropy & Information ----------------
    Entropy,
    CrossEntropy,
    KLDivergence,
    JSDivergence,
    MutualInformation,
    ConditionalEntropy,

    // ---------------- Characteristic Functions ----------------
    NormalCharacteristic,
    CharacteristicFromSamples,

    // ---------------- Time Series Statistics ----------------
    RollingMean,
    RollingVariance,
    EMA,
    Autocovariance,
    PartialAutocorrelation,
    HurstExponent,
    Detrend,
    Difference,
    LyapunovExponent,

    // ---------------- Sampling & Resampling ----------------
    BootstrapMean,
    BootstrapCI,
    Jackknife,
    PermutationTest,

    // ---------------- Regression & Estimation ----------------
    LinearRegression,
    PolynomialRegression,
    LeastSquares,

    // ---------------- Outliers & Anomaly ----------------
    ZScore,
    ModifiedZScore,
    GrubbsTest,
    ChauvenetCriterion,
    IsOutlier,

    _Count // всегда последний элемент, чтобы знать число функций
};

