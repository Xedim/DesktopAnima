//PatternList.h

X(Sqrt, UnaryReal, sqrt, "sqrt", Interval{0, INF}, Interval{0, INF}, PatternKind::Algebra)
X(Sign,      UnaryReal, sign);
X(Abs,       UnaryReal, abs);
X(Exp,       UnaryReal, exp);
X(Sin,       UnaryReal, sin);
X(Mean,      VecReal,   mean);
X(Variance,  VecReal,   variance);
X(Binomial, IntInt, binomial, "binomial", Interval{ 0, INF }, Interval{ 0, INF }, PatternKind::Algebra)
