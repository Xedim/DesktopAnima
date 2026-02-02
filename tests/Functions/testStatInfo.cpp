#include <gtest/gtest.h>
#include "../math/pattern/Functions.h"
#include <vector>
#include <cmath>

static VecReal normalize(VecReal v) {
    Real s = 0;
    for (Real x : v) s += x;
    for (Real& x : v) x /= s;
    return v;
}

static VecReal joint_normalize(const VecReal& px, const VecReal& py) {
    VecReal pxy(px.size() * py.size());
    for (size_t i = 0; i < px.size(); ++i)
        for (size_t j = 0; j < py.size(); ++j)
            pxy[i * py.size() + j] = px[i] * py[j];
    return pxy;
}

// ==========================================================
// ================= Entropy & Information ==================
// ==========================================================

TEST(Information, EntropyBasicProperties) {
    VecReal p = normalize({1,1,1,1});
    VecReal q = normalize({1,0,0,0});

    Real Hp = Functions::entropy(p);
    Real Hq = Functions::entropy(q);

    EXPECT_GE(Hp, 0);
    EXPECT_GE(Hq, 0);
    EXPECT_NEAR(Hq, 0.0, 1e-12);
    EXPECT_GT(Hp, Hq);
}

TEST(Information, CrossEntropySelf) {
    VecReal p = normalize({1,2,3,4});
    Real H = Functions::entropy(p);
    Real CE = Functions::cross_entropy(p, p);

    EXPECT_NEAR(H, CE, 1e-12);
}

TEST(Information, KLDivergenceProperties) {
    VecReal p = normalize({1,2,3});
    VecReal q = normalize({3,2,1});

    Real dpp = Functions::kl_divergence(p, p);
    Real dpq = Functions::kl_divergence(p, q);

    EXPECT_NEAR(dpp, 0.0, 1e-12);  // KL(p||p) = 0
    EXPECT_GE(dpq, 0.0);            // KL ≥ 0
}

TEST(Information, JSDivergenceProperties) {
    VecReal p = normalize({1,2,3});
    VecReal q = normalize({3,2,1});

    Real d1 = Functions::js_divergence(p, q);
    Real d2 = Functions::js_divergence(q, p);
    Real d0 = Functions::js_divergence(p, p);

    EXPECT_GE(d1, 0.0);
    EXPECT_NEAR(d1, d2, 1e-12);  // симметрия
    EXPECT_NEAR(d0, 0.0, 1e-12); // JS(p||p) = 0
}

TEST(Information, MutualInformationProperties) {
    VecReal px = normalize({1,2,3});
    VecReal py = normalize({3,2});
    VecReal pxy = joint_normalize(px, py);

    Real I = Functions::mutual_information(px, py, pxy);

    EXPECT_GE(I, 0.0);  // MI ≥ 0
    Real Hx = Functions::entropy(px);
    Real Hy = Functions::entropy(py);

    // MI ≤ min(Hx, Hy) для независимых переменных
    EXPECT_LE(I, std::min(Hx, Hy));
}

TEST(Information, ConditionalEntropyProperties) {
    VecReal px = normalize({1,2,3});
    VecReal py = normalize({3,2});
    VecReal pxy = joint_normalize(px, py);

    Real Hx_given_y = Functions::conditional_entropy(py, pxy);
    Real Hx = Functions::entropy(px);

    EXPECT_GE(Hx_given_y, 0.0);   // H(X|Y) ≥ 0
    EXPECT_LE(Hx_given_y, Hx);    // H(X|Y) ≤ H(X)
}
