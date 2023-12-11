[CSDN链接](https://blog.csdn.net/zzb714121/article/details/134921839?spm=1001.2014.3001.5502)，详情见csdn链接，我的这个csdn链接写的更加详细一些。
代码运行逻辑：
```
mkdir src
cd src
git clone "对应的git"
catkin_make
```
# 1  高斯牛顿法与gtsam学习心得
最近这一周一直在学习lio-sam，顺便把gtsam再学习下。学习流程主要是先学习了非线性优化部分，分别写了`gaussNewton.cpp`、`testGtsam.cpp`、`curveFitting.cpp`三个cpp。后面代码上传到我的GitHub上，目前还未上传，注意我的代码中依赖了`matplotlibcpp`，没有这个库的同学直接在头文件中删除它即可。
cpp名称     | 具体内容
-------- | -----
gaussNewton.cpp  | ****手写高斯牛顿法****（视觉slam十四讲书中的曲线拟合133页**曲线拟合指数函数**），参考了[b站讲解](https://www.bilibili.com/video/BV14D4y1A7Lj/?spm_id_from=333.999.0.0)，链接主要是帮助理解高斯牛顿增量方程中的累加，因为原始式子（1）是没有累加的，但是在实际计算时式子（2）进行了累加。
testGtsam.cpp  |**复现gtsam**官网中的`BetweenFactor`，特别是自定义因子`UnaryFactor`，参考[gtsam文档](https://github.com/borglab/gtsam/blob/develop/doc/gtsam.pdf)
 curveFitting.cpp | **用gtsam实现视觉slam十四讲133页的曲线拟合指数函数**，构造自定义因子，参考[gtsam曲线拟合](https://blog.csdn.net/weixin_41681988/article/details/132001320?spm=1001.2014.3001.5502)
 
 其中在写前两个cpp时没有遇到特别的问题，写`curveFitting.cpp`时遇到了一些问题，第一个是推导雅克比矩阵，在自定义因子的时候，需要推导出误差方程的雅克比矩阵，这部分遇到了比较多的的问题。第二个是代码运行过程中遇到的两个问题，这两个问题分别记录到下面1.3中.
## 1.1 手写高斯牛顿法 gaussNewton.cpp
重写高斯牛顿法
## 1.2  gtsam自定义UnaryFactor因子
在对二维刚体变换的推导过程中遇到了问题，因为自定义因子时，需要用到误差方程的雅克比矩阵，因此推导雅克比矩阵的时候遇到了问题，参考[gtsam曲线拟合](https://blog.csdn.net/weixin_41681988/article/details/132001320?spm=1001.2014.3001.5502)，这里面做了推导。
## 1.3 gtsam实现曲线拟合指数函数（视觉slam十四讲第二版133页）

这个cpp是`curveFitting.cpp`，其中结合了`matplotlibcpp.h`，`matplotlibcpp.h`是安装的c++画图的依赖库，方便画图使用，安装参考[链接](https://blog.csdn.net/kkbca/article/details/134421442)，这里面和1.1节中一样，都是对曲线进行拟合，因此它的雅克比矩阵也是现成的，具体可以参考下面代码中的`evaluateError`函数。先把我写的代码附上，如下所示：

```cpp
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
#include "matplotlibcpp.h"
#include <cmath>

#include <ros/ros.h>

using namespace std;
using namespace gtsam;

namespace plt = matplotlibcpp;
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
    // ros::init(argc, argv, "lio_sam");    

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

    plt::figure_size(640, 480);
    plt::plot(x_data, y_data, "ro");
    plt::plot(x, y, {{"color", "blue"}, {"label", "$y = e^{ax^2+bx+c}$"}});
    plt::show();

    // ros::MultiThreadedSpinner spinner(3);
    // spinner.spin();

    return 0;
}

```
最终拟合后的曲线如下图所示：
![拟合效果](https://img-blog.csdnimg.cn/direct/d4375feea7bf437cb66e1bf65361ce54.png#pic_center)
在写自定义因子的时候踩了两个坑，分别在1.31.和1.3.2中进行介绍。
### 1.3.1 Eigen返回错误
第一个错误是在上述代码中的 ` // return gtsam::Vector1(val - yi); // 返回值为残差`，这样写会返回错误，
`lio_sam_curveFitting: /usr/local/include/gtsam/3rdparty/Eigen/Eigen/src/Core/Matrix.h:241: Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>::Matrix(Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>::Index) [with _Scalar = double; int _Rows = 1; int _Cols = 1; int _Options = 0; int _MaxRows = 1; int _MaxCols = 1; Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>::Index = long int]: ' Assertion SizeAtCompileTime == Dynamic || SizeAtCompileTime == dim' failed.
[lio_sam_curveFitting-2] process has died [pid 1903900, exit code -6, cmd /home/zhao/Codes/competition_code/lio-sam-mergePoints/devel/lib/lio_sam/lio_sam_curveFitting __name:=lio_sam_curveFitting __log:=/home/zhao/.ros/log/46e8b8fa-9510-11ee-b1d4-65b78ef41fcf/lio_sam_curveFitting-2.log].
log file: /home/zhao/.ros/log/46e8b8fa-9510-11ee-b1d4-65b78ef41fcf/lio_sam_curveFitting-2*.log`
最后的解决方式在见[stackoverflow](https://stackoverflow.com/questions/77627134/gtsam-error-report-assertion-sizeatcompiletime-dynamic-sizeatcompiletim)，这个是我在stackoverflow提的问题，别人回答说主要是Eigen的版本太老了或者gtsam依赖的太老了。主要是更换了返回的方式，如下
```cpp
  gtsam::Vector1 ret; 
  ret[0] = val - yi;
  return ret;
```
### 1.3.2 ros的初始化干扰gtsam的优化
这个问题的主要干扰项是   `ros::init(argc, argv, "lio_sam");`，它会影响gtsam中的优化函数`optimize()`的使用，目前的解决方案是直接删除这一行代码，要不然会报错，报错如下：
![报错信息](https://img-blog.csdnimg.cn/direct/238b00130082469a8717fee85892f2bd.png#pic_center)
我在[stackoverflow](https://stackoverflow.com/questions/77633725/ros-initialization-will-affect-the-optimize-optimization-function-of-gtsam)提了问题，目前还没人回答。



