#include <gtsam/base/Vector.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/inference/Key.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <opencv2/core/core.hpp>
#include <random>
#include <cmath>
#include <cstdio>

#include <ros/ros.h>

// #include "matplotlibcpp.h"
// namespace plt = matplotlibcpp;

using namespace std;
using namespace gtsam;

using gtsam::symbol_shorthand::X;

// y = exp(a x^2 + b x + c)
// 利用x和参数a,b,c计算y
double funct(const gtsam::Vector3 &p, const double x)
{
    return exp(p(0) * x * x + p(1) * x + p(2));
}

// 自定义类名 : 继承于一元因子类<优化变量的数据类型>
class curvfitFactor : public gtsam::NoiseModelFactor1<gtsam::Vector3>
{
    double xi, yi; // 观测值

public:

    curvfitFactor(gtsam::Key j, const gtsam::SharedNoiseModel &model, double x, double y)
        : gtsam::NoiseModelFactor1<gtsam::Vector3>(model, j), xi(x), yi(y)  {}

    ~curvfitFactor() override {}

    // 自定义因子一定要重写evaluateError函数(优化变量, 雅可比矩阵)
    Vector evaluateError(const gtsam::Vector3 &p, boost::optional<Matrix &> H = boost::none) const override
    {
        auto val = funct(p, xi);
        if (H) // 残差为1维，优化变量为3维，雅可比矩阵为1*3
        {
            gtsam::Matrix Jac = gtsam::Matrix::Zero(1, 3);
            Jac << xi * xi * val, xi * val, val;
            (*H) = Jac;
        }
        gtsam::Vector1 ret; 
        ret[0] = val - yi;
        return ret;
        // return gtsam::Vector1(val - yi); // 返回值为残差
    }

};


int main(int argc, char** argv)
{
    // ros::init(argc, argv, "test_gtsam");    

    gtsam::NonlinearFactorGraph graph;
    const gtsam::Vector3 para(1.0, 2.0, 1.0); // a,b,c的真实值
    double w_sigma = 1.0;                     // 噪声Sigma值
    cv::RNG rng; // OpenCV随机数产生器

    std::vector<double> x_data, y_data;

    for (int i = 0; i < 100; ++i)
    {
        double xi = i / 100.0;

        double yi = funct(para, xi) + rng.gaussian(w_sigma * w_sigma); // 加入了噪声数据
        // auto noiseM = gtsam::noiseModel::Isotropic::Sigma(1, w_sigma); // 噪声的维度需要与观测值维度保持一致
        auto noiseM = gtsam::noiseModel::Diagonal::Sigmas(Vector1(w_sigma)); // 噪声的维度需要与观测值维度保持一致
        // 这里面的X(0)表示的是在第一个变量上加入一元因子
        graph.emplace_shared<curvfitFactor>(X(0), noiseM,xi, yi);     // 加入一元因子

        x_data.push_back(xi);
        y_data.push_back(yi);
    }

    gtsam::Values intial;
    intial.insert<gtsam::Vector3>(X(0), gtsam::Vector3(2.0, -1.0, 5.0));

    // gtsam::DoglegOptimizer opt(graph, intial); // 使用Dogleg优化
    gtsam::GaussNewtonOptimizer opt(graph, intial); // 使用高斯牛顿优化
    // gtsam::LevenbergMarquardtOptimizer opt(graph, intial); // 使用LM优化

    std::cout << "initial error=" << graph.error(intial) << std::endl;
    auto res = opt.optimize();
    res.print("final res:");

    std::cout << "final error=" << graph.error(res) << std::endl;

    gtsam::Vector3 matX0 = res.at<gtsam::Vector3>(X(0));
    std::cout << "a b c: " << matX0 << "\n";

    int n = 5000;
    std::vector<double> x(n), y(n), w(n, 2);
    for (int i = 0; i < n; ++i)
    {
        x.at(i) = i * 1.0 / n;
        y.at(i) = exp(matX0(0) * x[i] * x[i] + matX0(1) * x[i] + matX0(2));
    }
    
    getchar();
    // plt::figure_size(640, 480);
    // plt::plot(x_data, y_data, "ro");
    // plt::plot(x, y, {{"color", "blue"}, {"label", "$y = e^{ax^2+bx+c}$"}});
    // plt::show();

    return 0;
}
