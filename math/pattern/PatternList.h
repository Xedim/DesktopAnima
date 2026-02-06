//PatternList.h
#include "../math/common/Types.h"

// =================================================== Sign ============================================================
X(Sign,      sign,      "sign",       {ArgType::Real},       PatternKind::Sign)
X(Abs,       abs,       "abs",        {ArgType::Real},       PatternKind::Sign)
X(Heaviside, heaviside, "heaviside",  {ArgType::Real},       PatternKind::Sign)

// ================================================= Algebraic =========================================================
X(Factorial,      factorial,    "factorial",     {ArgType::Int},                                      PatternKind::Algebra)
X(Binomial,       binomial,     "binomial",      {ArgType::Int, ArgType::Int},                        PatternKind::Algebra)
X(Combination,    combination,  "combination",   {ArgType::Int, ArgType::Int},                        PatternKind::Algebra)
X(Permutation,    permutation,  "permutation",   {ArgType::Int, ArgType::Int},                        PatternKind::Algebra)
X(Mod,            mod,          "mod",           {ArgType::Real, ArgType::Real},                      PatternKind::Algebra)
X(Polynomial,     polynomial,   "polynomial",    {ArgType::Real, ArgType::VecReal},                   PatternKind::Algebra)
X(Rational,       rational,     "rational",      {ArgType::Real, ArgType::VecReal, ArgType::VecReal}, PatternKind::Algebra)

// ================================================ Power / Root =======================================================
X(Pow,            pow,            "pow",             {ArgType::Real, ArgType::Real},      PatternKind::Power)
X(XPowY,          x_pow_y,        "x_pow_y",         {ArgType::Real, ArgType::Real},      PatternKind::Power)
X(AlgebraicRoot,  algebraic_root, "algebraic_root",  {ArgType::Real, ArgType::VecReal},   PatternKind::Power)
X(Sqrt,           sqrt,           "sqrt",            {ArgType::Real},                     PatternKind::Power)
X(Sqrt1pM1,       sqrt1pm1,       "sqrt1pm1",        {ArgType::Real},                     PatternKind::Power)
X(Cbrt,           cbrt,           "cbrt",            {ArgType::Real},                     PatternKind::Power)

// ========================================== Exponential / Logarithmic ================================================
X(Exp,           exp,           "exp",         {ArgType::Real},                  PatternKind::ExpLog)
X(Exp2,          exp2,          "exp2",        {ArgType::Real},                  PatternKind::ExpLog)
X(Expm1Safe,     expm1_safe,    "expm1_safe",  {ArgType::Real},                  PatternKind::ExpLog)
X(Log2,          log2,          "log2",        {ArgType::Real},                  PatternKind::ExpLog)
X(Log,           log,           "log",         {ArgType::Real},                  PatternKind::ExpLog)
X(Log10,         log10,         "log10",       {ArgType::Real},                  PatternKind::ExpLog)
X(LogA,          log_a,         "log_a",       {ArgType::Real, ArgType::Real},   PatternKind::ExpLog)
X(Log1p,         log1p,         "log1p",       {ArgType::Real},                  PatternKind::ExpLog)

// ================================================ Trigonometric ======================================================
X(Sin,           sin,           "sin",         {ArgType::Real},                  PatternKind::Trigonometric)
X(Cos,           cos,           "cos",         {ArgType::Real},                  PatternKind::Trigonometric)
X(Sec,           sec,           "sec",         {ArgType::Real},                  PatternKind::Trigonometric)
X(Csc,           csc,           "csc",         {ArgType::Real},                  PatternKind::Trigonometric)
X(Sinc,          sinc,          "sinc",        {ArgType::Real},                  PatternKind::Trigonometric)
X(Tan,           tan,           "tan",         {ArgType::Real},                  PatternKind::Trigonometric)
X(Cot,           cot,           "cot",         {ArgType::Real},                  PatternKind::Trigonometric)
X(Asin,          asin,          "asin",        {ArgType::Real},                  PatternKind::Trigonometric)
X(Acos,          acos,          "acos",        {ArgType::Real},                  PatternKind::Trigonometric)
X(Atan,          atan,          "atan",        {ArgType::Real},                  PatternKind::Trigonometric)
X(Atan2,         atan2,         "atan2",       {ArgType::Real, ArgType::Real},   PatternKind::Trigonometric)
X(Hypot,         hypot,         "hypot",       {ArgType::Real, ArgType::Real},   PatternKind::Trigonometric)

// ================================================ Hyperbolic =========================================================
X(Sinh,           sinh,           "sinh",          {ArgType::Real},                 PatternKind::Hyperbolic)
X(Cosh,           cosh,           "cosh",          {ArgType::Real},                 PatternKind::Hyperbolic)
X(Sech,           sech,           "sech",          {ArgType::Real},                 PatternKind::Hyperbolic)
X(Csch,           csch,           "csch",          {ArgType::Real},                 PatternKind::Hyperbolic)
X(Tanh,           tanh,           "tanh",          {ArgType::Real},                 PatternKind::Hyperbolic)
X(Coth,           coth,           "coth",          {ArgType::Real},                 PatternKind::Hyperbolic)
X(Asinh,          asinh,          "asinh",         {ArgType::Real},                 PatternKind::Hyperbolic)
X(Acosh,          acosh,          "acosh",         {ArgType::Real},                 PatternKind::Hyperbolic)
X(Atanh,          atanh,          "atanh",         {ArgType::Real},                 PatternKind::Hyperbolic)

// ================================================= Special ===========================================================
X(Erf,            erf,            "erf",           {ArgType::Real},                             PatternKind::Special)
X(Erfc,           erfc,           "erfc",          {ArgType::Real},                             PatternKind::Special)
X(Gamma,          gamma,          "gamma",         {ArgType::Real},                             PatternKind::Special)
X(LGamma,         lgamma,         "lgamma",        {ArgType::Real},                             PatternKind::Special)
X(Beta,           beta,           "beta",          {ArgType::Real, ArgType::Real},              PatternKind::Special)

X(CylBesselJ,     cyl_bessel_j,   "cyl_bessel_j",  {ArgType::Real, ArgType::Real},              PatternKind::Special)
X(CylNeumann,     cyl_neumann,    "cyl_neumann",   {ArgType::Real, ArgType::Real},              PatternKind::Special)
X(CylBesselI,     cyl_bessel_i,   "cyl_bessel_i",  {ArgType::Real, ArgType::Real},              PatternKind::Special)
X(CylBesselK,     cyl_bessel_k,   "cyl_bessel_k",  {ArgType::Real, ArgType::Real},              PatternKind::Special)

X(LambertW,       lambert_w,      "lambert_w",     {ArgType::Real},                             PatternKind::Special)
X(Legendre,       legendre,       "legendre",      {ArgType::Int, ArgType::Real},               PatternKind::Special)
X(AssocLegendre,  assoc_legendre, "assoc_legendre",{ArgType::Int, ArgType::Int, ArgType::Real}, PatternKind::Special)
X(RiemannZeta,    riemann_zeta,   "riemann_zeta",  {ArgType::Real},                             PatternKind::Special)
X(Zeta,           zeta,           "zeta",          {ArgType::Real},                             PatternKind::Special)

// ========================================= Generalized ===============================================================
X(DiracDelta,      dirac_delta,       "dirac_delta",   {ArgType::Real},                     PatternKind::Generalized)
X(GeometricSum,    geometric_sum,     "geometric_sum", {ArgType::Real, ArgType::Int},       PatternKind::Generalized)

// ========================================== Numerical ================================================================
X(Round,           round,             "round",         {ArgType::Real},                                 PatternKind::Numerical)
X(Floor,           floor,             "floor",         {ArgType::Real},                                 PatternKind::Numerical)
X(Ceil,            ceil,              "ceil",          {ArgType::Real},                                 PatternKind::Numerical)
X(Trunc,           trunc,             "trunc",         {ArgType::Real},                                 PatternKind::Numerical)
X(Clamp,           clamp,             "clamp",         {ArgType::Real, ArgType::Real, ArgType::Real},   PatternKind::Numerical)
X(Lerp,            lerp,              "lerp",          {ArgType::Real, ArgType::Real, ArgType::Real},   PatternKind::Numerical)
X(FMA,             fma,               "fma",           {ArgType::Real, ArgType::Real, ArgType::Real},   PatternKind::Numerical)

// =========================================== Fractals ================================================================
X(Weierstrass,     weierstrass,       "weierstrass",   {ArgType::Real},                                                    PatternKind::Fractal)
X(Cantor,          cantor,            "cantor",        {ArgType::Real},                                                    PatternKind::Fractal)
X(Logistic,        logistic,          "logistic",      {ArgType::Real, ArgType::Real, ArgType::Int},                       PatternKind::Fractal)
X(Tent,            tent,              "tent",          {ArgType::Real, ArgType::Int},                                      PatternKind::Fractal)
X(Julia,           julia,             "julia",         {ArgType::Complex, ArgType::Complex, ArgType::Int},                 PatternKind::Fractal)
X(Escapes,         escapes,           "escapes",       {ArgType::Complex, ArgType::Complex, ArgType::Int, ArgType::Real},  PatternKind::Fractal)

// =========================================== Iteration ===============================================================
X(Iterate,         iterate,           "iterate",       {ArgType::Real, ArgType::Real, ArgType::Int},       PatternKind::Iteration)

// ====================================== Descriptive Statistics =======================================================
X(Sum,             sum,                       "sum",                        {ArgType::VecReal},            PatternKind::Statistical)
X(Mean,            mean,                      "mean",                       {ArgType::VecReal},            PatternKind::Statistical)
X(Median,          median,                    "median",                     {ArgType::VecReal},            PatternKind::Statistical)
X(Mode,            mode,                      "mode",                       {ArgType::VecReal},            PatternKind::Statistical)
X(Min,             min,                       "min",                        {ArgType::VecReal},            PatternKind::Statistical)
X(Max,             max,                       "max",                        {ArgType::VecReal},            PatternKind::Statistical)
X(Range,           range,                     "range",                      {ArgType::VecReal},            PatternKind::Statistical)
X(Variance,        variance,                  "variance",                   {ArgType::VecReal},            PatternKind::Statistical)
X(VarianceUnbiased,variance_unbiased,         "variance_unbiased",          {ArgType::VecReal},            PatternKind::Statistical)
X(StdDev,          stddev,                    "stddev",                     {ArgType::VecReal},            PatternKind::Statistical)
X(StdDevUnbiased,  stddev_unbiased,           "stddev_unbiased",            {ArgType::VecReal},            PatternKind::Statistical)
X(MAD,             mean_absolute_deviation,   "mean_absolute_deviation",    {ArgType::VecReal},            PatternKind::Statistical)

// ======================================== Shape Statistics ===========================================================
X(Skewness,        skewness,              "skewness",              {ArgType::VecReal},                  PatternKind::Statistical)
X(Kurtosis,        kurtosis,              "kurtosis",              {ArgType::VecReal},                  PatternKind::Statistical)
X(Moment,          moment,                "moment",                {ArgType::VecReal, ArgType::Int},    PatternKind::Statistical)
X(RawMoment,       raw_moment,            "raw_moment",            {ArgType::VecReal, ArgType::Int},    PatternKind::Statistical)

// ======================================== Order & Quantiles ==========================================================
X(Quantile,        quantile,              "quantile",              {ArgType::VecReal, ArgType::Real},   PatternKind::Statistical)
X(Percentile,      percentile,            "percentile",            {ArgType::VecReal, ArgType::Real},   PatternKind::Statistical)
X(Quartiles,       quartiles,             "quartiles",             {ArgType::VecReal},                  PatternKind::Statistical)
X(IQR,             iqr,                   "iqr",                   {ArgType::VecReal},                  PatternKind::Statistical)
X(TrimmedMean,     trimmed_mean,          "trimmed_mean",          {ArgType::VecReal, ArgType::Real},   PatternKind::Statistical)

// ======================================== Robust Statistics ==========================================================
X(MedianAbsDev,     median_absolute_deviation, "median_absolute_deviation",  {ArgType::VecReal},                  PatternKind::Statistical)
X(WinsorizedMean,   winsorized_mean,           "winsorized_mean",            {ArgType::VecReal, ArgType::Real},   PatternKind::Statistical)
X(HuberMean,        huber_mean,                "huber_mean",                 {ArgType::VecReal, ArgType::Real},   PatternKind::Statistical)
X(BiweightMean,     biweight_mean,             "biweight_mean",              {ArgType::VecReal},                  PatternKind::Statistical)
X(SNR,              snr,                       "snr",                        {ArgType::VecReal},                  PatternKind::Statistical)

// ==================================== Correlation & Dependence =======================================================
X(Covariance,       covariance,                "covariance",                 {ArgType::VecReal, ArgType::VecReal},               PatternKind::Statistical)
X(CorrelPearson,    correlation_pearson,       "correlation_pearson",        {ArgType::VecReal, ArgType::VecReal},               PatternKind::Statistical)
X(CorrelSpearman,   correlation_spearman,      "correlation_spearman",       {ArgType::VecReal, ArgType::VecReal},               PatternKind::Statistical)
X(CorrelKendall,    correlation_kendall,       "correlation_kendall",        {ArgType::VecReal, ArgType::VecReal},               PatternKind::Statistical)
X(Autocorrel,       autocorrelation,           "autocorrelation",            {ArgType::VecReal, ArgType::Int},                   PatternKind::Statistical)
X(CrossCorrel,      cross_correlation,         "cross_correlation",          {ArgType::VecReal, ArgType::VecReal, ArgType::Int}, PatternKind::Statistical)

// ==================================== Probability Distributions ======================================================
X(NormPDF,          dist::pdf,               "normal_pdf",                 {ArgType::DistNormal, ArgType::Real},         PatternKind::Distributional)
X(NormCDF,          dist::cdf,               "normal_cdf",                 {ArgType::DistNormal, ArgType::Real},         PatternKind::Distributional)
X(NormQuantile,     dist::quantile,          "normal_quantile",            {ArgType::DistNormal, ArgType::Real},         PatternKind::Distributional)
X(NormLogLike,      dist::log_likelihood,    "normal_log_likelihood",      {ArgType::DistNormal, ArgType::VecReal},      PatternKind::Distributional)

X(LogNormPDF,       dist::pdf,               "lognormal_pdf",              {ArgType::DistLogNormal, ArgType::Real},      PatternKind::Distributional)
X(LogNormCDF,       dist::cdf,               "lognormal_cdf",              {ArgType::DistLogNormal, ArgType::Real},      PatternKind::Distributional)
X(LogNormQuantile,  dist::quantile,          "lognormal_quantile",         {ArgType::DistLogNormal, ArgType::Real},      PatternKind::Distributional)
X(LogNormLogLike,   dist::log_likelihood,    "lognormal_log_likelihood",   {ArgType::DistLogNormal, ArgType::VecReal},   PatternKind::Distributional)

X(ExponentPDF,      dist::pdf,               "exponential_pdf",            {ArgType::DistExp, ArgType::Real},    PatternKind::Distributional)
X(ExponentCDF,      dist::cdf,               "exponential_cdf",            {ArgType::DistExp, ArgType::Real},    PatternKind::Distributional)
X(ExponentQuantile, dist::quantile,          "exponential_quantile",       {ArgType::DistExp, ArgType::Real},    PatternKind::Distributional)
X(ExponentLogLike,  dist::log_likelihood,    "exponential_log_likelihood", {ArgType::DistExp, ArgType::VecReal}, PatternKind::Distributional)

X(GammaPDF,         dist::pdf,               "gamma_pdf",                  {ArgType::DistGamma, ArgType::Real},          PatternKind::Distributional)
X(GammaCDF,         dist::cdf,               "gamma_cdf",                  {ArgType::DistGamma, ArgType::Real},          PatternKind::Distributional)
X(GammaQuantile,    dist::quantile,          "gamma_quantile",             {ArgType::DistGamma, ArgType::Real},          PatternKind::Distributional)
X(GammaLogLike,     dist::log_likelihood,    "gamma_log_likelihood",       {ArgType::DistGamma, ArgType::VecReal},       PatternKind::Distributional)

X(BetaPDF,          dist::pdf,               "beta_pdf",                   {ArgType::DistBeta, ArgType::Real},           PatternKind::Distributional)
X(BetaCDF,          dist::cdf,               "beta_cdf",                   {ArgType::DistBeta, ArgType::Real},           PatternKind::Distributional)
X(BetaQuantile,     dist::quantile,          "beta_quantile",              {ArgType::DistBeta, ArgType::Real},           PatternKind::Distributional)
X(BetaLogLike,      dist::log_likelihood,    "beta_log_likelihood",        {ArgType::DistBeta, ArgType::VecReal},        PatternKind::Distributional)

X(WeibullPDF,       dist::pdf,               "weibull_pdf",                {ArgType::DistWeibull, ArgType::Real},        PatternKind::Distributional)
X(WeibullCDF,       dist::cdf,               "weibull_cdf",                {ArgType::DistWeibull, ArgType::Real},        PatternKind::Distributional)
X(WeibullQuantile,  dist::quantile,          "weibull_quantile",           {ArgType::DistWeibull, ArgType::Real},        PatternKind::Distributional)
X(WeibullLogLike,   dist::log_likelihood,    "weibull_log_likelihood",     {ArgType::DistWeibull, ArgType::VecReal},     PatternKind::Distributional)

X(CauchyPDF,        dist::pdf,               "cauchy_pdf",                 {ArgType::DistCauchy, ArgType::Real},         PatternKind::Distributional)
X(CauchyCDF,        dist::cdf,               "cauchy_cdf",                 {ArgType::DistCauchy, ArgType::Real},         PatternKind::Distributional)
X(CauchyQuantile,   dist::quantile,          "cauchy_quantile",            {ArgType::DistCauchy, ArgType::Real},         PatternKind::Distributional)
X(CauchyLogLike,    dist::log_likelihood,    "cauchy_log_likelihood",      {ArgType::DistCauchy, ArgType::VecReal},      PatternKind::Distributional)

X(StudentTPDF,      dist::pdf,               "student_t_pdf",              {ArgType::DistStudentT, ArgType::Real},       PatternKind::Distributional)
X(StudentTCDF,      dist::cdf,               "student_t_cdf",              {ArgType::DistStudentT, ArgType::Real},       PatternKind::Distributional)
X(StudentTQuantile, dist::quantile,          "student_t_quantile",         {ArgType::DistStudentT, ArgType::Real},       PatternKind::Distributional)
X(StudentTLogLike,  dist::log_likelihood,    "student_t_log_likelihood",   {ArgType::DistStudentT, ArgType::VecReal},    PatternKind::Distributional)

// ======================================= Statistical Tests ===========================================================
X(ZTest,            z_test,                    "z_test",                     {ArgType::VecReal, ArgType::Real, ArgType::Real},       PatternKind::StatTest)
X(TTest,            t_test,                    "t_test",                     {ArgType::VecReal, ArgType::Real},                      PatternKind::StatTest)
X(WelchTTest,       welch_t_test,              "welch_t_test",               {ArgType::VecReal, ArgType::VecReal},                   PatternKind::StatTest)
X(MannWhitneyU,     mann_whitney_u,            "mann_whitney_u",             {ArgType::VecReal, ArgType::VecReal},                   PatternKind::StatTest)
X(WilcoxonSignRank, wilcoxon_signed_rank,      "wilcoxon_signed_rank",       {ArgType::VecReal, ArgType::VecReal},                   PatternKind::StatTest)
X(KSTest,           ks_test,                   "ks_test",                    {ArgType::VecReal, ArgType::VecReal},                   PatternKind::StatTest)
X(ChiSquareTest,    chi_square_test,           "chi_square_test",            {ArgType::VecReal, ArgType::VecReal},                   PatternKind::StatTest)
X(AndersonDarling,  anderson_darling,          "anderson_darling",           {ArgType::VecReal},                                     PatternKind::StatTest)

// ====================================== Entropy & Information ========================================================
X(Entropy,          entropy,                   "entropy",                    {ArgType::VecReal},                                     PatternKind::Information)
X(CrossEntropy,     cross_entropy,             "cross_entropy",              {ArgType::VecReal, ArgType::VecReal},                   PatternKind::Information)
X(KLDivergence,     kl_divergence,             "kl_divergence",              {ArgType::VecReal, ArgType::VecReal},                   PatternKind::Information)
X(JSDivergence,     js_divergence,             "js_divergence",              {ArgType::VecReal, ArgType::VecReal},                   PatternKind::Information)
X(MutualInfo,       mutual_information,        "mutual_information",         {ArgType::VecReal, ArgType::VecReal, ArgType::VecReal}, PatternKind::Information)
X(ConditEntropy,    conditional_entropy,       "conditional_entropy",        {ArgType::VecReal, ArgType::VecReal},                   PatternKind::Information)

// ===================================== Characteristic Functions ======================================================
X(NormCharact,      normal_characteristic,     "normal_characteristic",      {ArgType::Real, ArgType::Real, ArgType::Real},     PatternKind::Characteristical)
X(SamplesCharact,   samples_characteristic,    "samples_characteristic",     {ArgType::VecReal, ArgType::Real},                 PatternKind::Characteristical)

// ====================================== Time Series Statistics =======================================================
X(RollingMean,      rolling_mean,              "rolling_mean",              {ArgType::VecReal, ArgType::SizeT},              PatternKind::TimeSeries)
X(RollingVariance,  rolling_variance,          "rolling_variance",          {ArgType::VecReal, ArgType::SizeT},              PatternKind::TimeSeries)
X(EMA,              ema,                       "ema",                        {ArgType::VecReal, ArgType::Real},              PatternKind::TimeSeries)
X(Autocovariance,   autocovariance,            "autocovariance",            {ArgType::VecReal, ArgType::Int},                PatternKind::TimeSeries)
X(PartAutocorrel,   partial_autocorrelation,   "partial_autocorrelation",   {ArgType::VecReal, ArgType::Int},                PatternKind::TimeSeries)
X(HurstExponent,    hurst_exponent,            "hurst_exponent",            {ArgType::VecReal},                              PatternKind::TimeSeries)
X(Detrend,          detrend,                   "detrend",                   {ArgType::VecReal},                              PatternKind::TimeSeries)
X(Difference,       difference,                "difference",                {ArgType::VecReal, ArgType::Int},                PatternKind::TimeSeries)
X(LyapunovExponent, lyapunov_exponent,         "lyapunov_exponent",         {ArgType::VecReal},                              PatternKind::TimeSeries)
X(TakensMap,        takens_map,                "takens_map",                {ArgType::VecReal, ArgType::Int, ArgType::Int},  PatternKind::TimeSeries)

// ======================================= Sampling & Resampling =======================================================
X(BootstrapMean,    bootstrap_mean,            "bootstrap_mean",            {ArgType::VecReal, ArgType::Int},                    PatternKind::Resampling)
X(BootstrapCI,      bootstrap_ci,              "bootstrap_ci",              {ArgType::VecReal, ArgType::Real, ArgType::Int},     PatternKind::Resampling)
X(Jackknife,        jackknife,                 "jackknife",                 {ArgType::VecReal},                                  PatternKind::Resampling)
X(PermutationTest,  permutation_test,          "permutation_test",          {ArgType::VecReal, ArgType::VecReal, ArgType::Int},  PatternKind::Resampling)

// ===================================== Regression & Estimation =======================================================
X(LinearRegression, linear_regression,         "linear_regression",         {ArgType::VecReal, ArgType::VecReal},                PatternKind::Regression)
X(PolyRegression,   polynomial_regression,     "polynomial_regression",     {ArgType::VecReal, ArgType::VecReal, ArgType::Int},  PatternKind::Regression)
X(LeastSquares,     least_squares,             "least_squares",             {ArgType::VecReal},                                  PatternKind::Regression)

// ======================================= Outliers & Anomaly ==========================================================
X(ZScore,           z_score,                   "z_score",             {ArgType::VecReal},                                           PatternKind::Outliers)
X(ModifiedZScore,   modified_z_score,          "modified_z_score",    {ArgType::VecReal},                                           PatternKind::Outliers)
X(GrubbsTest,       grubbs_test,               "grubbs_test",        {ArgType::VecReal, ArgType::Real},                             PatternKind::Outliers)
X(ChauvenetCrit,    chauvenet_criterion,       "chauvenet_criterion",{ArgType::VecReal},                                            PatternKind::Outliers)
X(IsOutlier,        is_outlier,                "is_outlier",         {ArgType::Real, ArgType::Real, ArgType::Real, ArgType::Real},  PatternKind::Outliers)
