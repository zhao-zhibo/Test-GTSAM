
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/Scenario.h>
#include "utility.h"

// #include "matplotlibcpp.h"
// namespace plt = matplotlibcpp;

using namespace std;
using namespace gtsam;

// 类名 : 继承于NoiseModelFactor1<优化变量类型>  3.2 Defining Custom Factors
class UnaryFactor : public NoiseModelFactor1<Pose2>
{
    // 观测值(x,y),当然也可以采用Point2
    double mx_, my_;

public:
    // 需要重写这个函数来定义误差和雅可比矩阵
    using NoiseModelFactor1<Pose2>::evaluateError;

    // 因子的智能指针
    typedef std::shared_ptr<UnaryFactor> shared_ptr;

    // 一元因子的构造函数， (key , 测量值， 噪声模型)
    UnaryFactor(Key j, double x, double y, const SharedNoiseModel &model) : NoiseModelFactor1<Pose2>(model, j), mx_(x), my_(y) {}

    ~UnaryFactor() override {}

    Vector evaluateError(const Pose2 &q, boost::optional<Matrix &> H = boost::none) const override
    {
       
        // 雅可比矩阵，在切空间等于右手定则的旋转矩阵
        // H =  [ cos(q.theta)  -sin(q.theta) 0 ]
        //      [ sin(q.theta)   cos(q.theta) 0 ]
        // std::cout << "UnaryFactor 3:" << std::endl;
        const Rot2 &R = q.rotation();
        if (H)
        {
            // std::cout << "UnaryFactor 3.1:" << std::endl;
            (*H) = (gtsam::Matrix(2, 3) << R.c(), -R.s(), 0.0, R.s(), R.c(), 0.0).finished();
        }

        // 返回误差
        // std::cout << "UnaryFactor 6:" << std::endl;
        return (Vector(2) << q.x() - mx_, q.y() - my_).finished();
    }

    // The second is a 'clone' function that allows the factor to be copied. Under most
    // circumstances, the following code that employs the default copy constructor should
    // work fine.
    gtsam::NonlinearFactor::shared_ptr clone() const override
    {
        return boost::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new UnaryFactor(*this)));
    }

};

/*******************************Velodyne点云结构**********************************/
struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D     // 位置
    PCL_ADD_INTENSITY;  // 激光点反射强度，也可以存点的索引
    uint16_t ring;      // 扫描线
    float time;         // 时间戳，记录相对于当前帧第一个激光点的时差，第一个点time=0
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (std::uint16_t, ring, ring) (float, time, time)
)

// Use the Velodyne point format as a common representation
using PointXYZIRT = VelodynePointXYZIRT;
// imu数据队列长度
const int queueLength = 2000;

class testGtsam : public ParamServer
{
private:
    // imu队列、odom队列互斥锁
    std::mutex imuLock;
    std::mutex odoLock;
    std::mutex veloLock;
    // 订阅原始激光点云
    //ros::Subscriber subLaserCloud;
    ros::Subscriber subLaserCloudfront;
    ros::Subscriber subLaserCloudrear;
    ros::Publisher  pubfullCloud; //发布初始合并点云

    ros::Publisher  pubLaserCloud;
    // 发布当前帧校正后点云，有效点
    ros::Publisher pubExtractedCloud;
    ros::Publisher pubLaserCloudInfo;
    // imu数据队列（原始数据，转lidar系下）

    // 激光点云数据队列
    std::deque<sensor_msgs::PointCloud2> cloudQueue;
    std::deque<sensor_msgs::PointCloud2> cachePointCloudfrontQueue;
    std::deque<sensor_msgs::PointCloud2> cachePointCloudrearQueue;
    std::deque<pcl::PointCloud<PointXYZIRT>::Ptr> pointCloudfrontQueue;
    std::deque<pcl::PointCloud<PointXYZIRT>::Ptr> pointCloudrearQueue;
    // 队列front帧，作为当前处理帧点云
    sensor_msgs::PointCloud2 currentCloudMsg;

    sensor_msgs::PointCloud2 currentPointCloudfrontMsg;
    sensor_msgs::PointCloud2 currentPointCloudrearMsg;

    pcl::PointCloud<PointXYZIRT>::Ptr pointCloudfrontIn;
    pcl::PointCloud<PointXYZIRT>::Ptr pointCloudrearIn;

    pcl::PointCloud<PointXYZIRT>::Ptr pointCloudfront;
    pcl::PointCloud<PointXYZIRT>::Ptr pointCloudrear;
    pcl::PointCloud<PointXYZIRT>::Ptr pointCloudFull;


    int deskewFlag;
   
    std_msgs::Header cloudHeader;

    vector<int> columnIdnCountVec;


public:
    testGtsam():
    deskewFlag(0)
    {   // 订阅原始imu数据

        // pcl日志级别，只打ERROR日志
        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }
   /***************分配内存（初始化被调用）***************/
   

    ~testGtsam(){}

    void TestOdoFactor()
    {
        // gtsam.pdf 2.1~2.5
        // Create an empty nonlinear factor graph 
        gtsam::NonlinearFactorGraph graph;

        // Add a Gaussian prior on pose x_1
        gtsam::Pose2 priorMean(0.0, 0.0, 0.0); 
        auto priorNoise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(0.3, 0.3, 0.1)); 
        graph.add(gtsam::PriorFactor<gtsam::Pose2>(1, priorMean, priorNoise));
           
        // 加入两个里程计因子
        gtsam::Pose2 odometry(2.0, 0.0, 0.0); 
        auto odometryNoise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(0.2, 0.2, 0.1)); 
        graph.add(gtsam::BetweenFactor<gtsam::Pose2>(1, 2, odometry, odometryNoise)); 
        graph.add(gtsam::BetweenFactor<gtsam::Pose2>(2, 3, odometry, odometryNoise));


        // 3.3. Using Custom Factors  加入自定义的一元因子测量值
        auto unaryNoise = noiseModel::Diagonal::Sigmas(Vector2(0.1, 0.1)); // 其中需要噪声模型维度需要和测量值的维度保持一致，在XY方向标准差为10cm// 10cm std on x,y
        graph.emplace_shared<UnaryFactor>(1, 0.0, 0.0, unaryNoise);
        graph.emplace_shared<UnaryFactor>(2, 2.0, 0.0, unaryNoise);
        graph.emplace_shared<UnaryFactor>(3, 4.0, 0.0, unaryNoise);

        gtsam::Values initial;
        initial.insert(1, gtsam::Pose2(0.5, 0.0, 0.2));
        initial.insert(2, gtsam::Pose2(2.3, 0.1, -0.2));
        initial.insert(3, gtsam::Pose2(4.1, 0.1, 0.1));

        gtsam::Values result = gtsam::LevenbergMarquardtOptimizer(graph, initial).optimize();
        result.print();

        std::cout.precision(2); 
        gtsam::Marginals marginals(graph, result); 
        std::cout << "x1 covariance:\n" << marginals.marginalCovariance(1) << std::endl; 
        std::cout << "x2 covariance:\n" << marginals.marginalCovariance(2) << std::endl; 
        std::cout << "x3 covariance:\n" << marginals.marginalCovariance(3) << std::endl;

    }


    void LoopClosure()
    {
        gtsam::NonlinearFactorGraph graph;
        auto priorNoise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(0.3, 0.3, 0.1));
        // graph.addPrior(1, gtsam::Pose2(0,0,0), priorNoise); addPrior函数已经被删除了，用add函数即可
        graph.add(gtsam::PriorFactor<gtsam::Pose2>(1, gtsam::Pose2(0,0,0), priorNoise));

        // 加入里程计因子
        auto odometryNoise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(0.2, 0.2, 0.1));
        // 正常里程计往前走，分别加入 1 2 3 4 5之间的约束
        graph.add(gtsam::BetweenFactor<gtsam::Pose2>(1, 2, gtsam::Pose2(2, 0, 0 ), odometryNoise));
        graph.add(gtsam::BetweenFactor<gtsam::Pose2>(2, 3, gtsam::Pose2(2, 0, M_PI_2), odometryNoise));
        graph.add(gtsam::BetweenFactor<gtsam::Pose2>(3, 4, gtsam::Pose2(2, 0, M_PI_2), odometryNoise));
        graph.add(gtsam::BetweenFactor<gtsam::Pose2>(4, 5, gtsam::Pose2(2, 0, M_PI_2), odometryNoise));

        // 加入了回环约束
        graph.add(gtsam::BetweenFactor<gtsam::Pose2>(5, 2, gtsam::Pose2(2, 0, M_PI_2), odometryNoise));
        
    }

};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "test_gtsam");

    testGtsam IP;
    ROS_INFO("\033[1;32m----> testGtsam Started.\033[0m");

    IP.TestOdoFactor();
   
    //三个线程
    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();
    
    return 0;
}
