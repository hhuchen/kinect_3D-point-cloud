#include <iostream>
#include <vector>

// OpenCV Header
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// OpenNI Header
#include <OpenNI.h>

// PCL 
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>  //体素滤波相关
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>




typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

// namespace
using namespace cv;
using namespace std;
using namespace openni;


//相机参数
struct CAMERA_INTRINSIC_PARAMETERS
{
	double cx;
	double cy;
	double fx;
	double fy;
	double scale;
};

// image2PonitCloud 将rgb图转换为点云
pcl::PointCloud<pcl::PointXYZ>::Ptr image2PointCloud(cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera);
// input: 3维点Point3f (u,v,d)// point2dTo3d 将单个点从图像坐标转换为空间坐标
cv::Point3f point2dTo3d(cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera);
//相机投影函数
cv::Point2f point3dTo2d(cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera);
// 行中值高斯自适应滤波
cv::Mat RM_Gaussian_filter(cv::Mat img_depth);
// 过滤掉1.5米外所有的深度点
cv::Mat remove_over(cv::Mat depth);
//复现的论文的算法
cv::Mat paper_way(pcl::PointCloud<pcl::PointXYZ>::Ptr src, cv::Mat img, CAMERA_INTRINSIC_PARAMETERS& camera);


void viewerOneOff(pcl::visualization::PCLVisualizer& viewer)
{
	viewer.setBackgroundColor(0, 0, 0);//设置背景颜色 
}


int main(int argc, char **argv)
{
	// 1. Initial OpenNI
	if (OpenNI::initialize() != STATUS_OK)
	{
		cerr << "OpenNI Initial Error: "
			<< OpenNI::getExtendedError() << endl;
		return -1;
	}

	// 2. Open Device
	Device mDevice;
	if (mDevice.open(ANY_DEVICE) != STATUS_OK)
	{
		cerr << "Can't Open Device: "
			<< OpenNI::getExtendedError() << endl;
		return -1;
	}
	openni::Status status;

	CAMERA_INTRINSIC_PARAMETERS camera;
	camera.cx = 318.1;
	camera.cy = 232.8;
	camera.fx = 523.9;
	camera.fy = 524.2;
	camera.scale = 1000.0;

	// 3. Create depth stream
	VideoStream mDepthStream;
	if (mDevice.hasSensor(SENSOR_DEPTH))
	{
		if (mDepthStream.create(mDevice, SENSOR_DEPTH) == STATUS_OK)
		{
			// 3a. set video mode
			VideoMode mMode;
			mMode.setResolution(640, 480);
			mMode.setFps(30);
			mMode.setPixelFormat(PIXEL_FORMAT_DEPTH_1_MM);

			if (mDepthStream.setVideoMode(mMode) != STATUS_OK)
			{
				cout << "Can't apply VideoMode: "
					<< OpenNI::getExtendedError() << endl;
			}
		}
		else
		{
			cerr << "Can't create depth stream on device: "
				<< OpenNI::getExtendedError() << endl;
			return -1;
		}
	}
	else
	{
		cerr << "ERROR: This device does not have depth sensor" << endl;
		return -1;
	}

	// 4. Create color stream
	VideoStream mColorStream;
	if (mDevice.hasSensor(SENSOR_COLOR))
	{
		if (mColorStream.create(mDevice, SENSOR_COLOR) == STATUS_OK)
		{
			// 4a. set video mode
			VideoMode mMode;
			mMode.setResolution(640, 480);
			mMode.setFps(30);
			mMode.setPixelFormat(PIXEL_FORMAT_RGB888);

			if (mColorStream.setVideoMode(mMode) != STATUS_OK)
			{
				cout << "Can't apply VideoMode: "
					<< OpenNI::getExtendedError() << endl;
			}

			// 4b. image registration
			if (mDevice.isImageRegistrationModeSupported(
				IMAGE_REGISTRATION_DEPTH_TO_COLOR))
			{
				mDevice.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);
			}
		}
		else
		{
			cerr << "Can't create color stream on device: "
				<< OpenNI::getExtendedError() << endl;
			return -1;
		}
	}

	if (mDevice.isImageRegistrationModeSupported(IMAGE_REGISTRATION_DEPTH_TO_COLOR))
	{
		status = mDevice.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);
	}
	// 5. create OpenCV Window
	cv::namedWindow("Depth Image", CV_WINDOW_AUTOSIZE);
	cv::namedWindow("Color Image", CV_WINDOW_AUTOSIZE);
	// 5. create PCL Window
	pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
	boost::shared_ptr<pcl::visualization::PCLVisualizer> view(new pcl::visualization::PCLVisualizer);
	//viewer.runOnVisualizationThreadOnce(viewerOneOff);

	string outputcolorPath = "colorout.avi";
	string outputdepthPath = "depthout.avi";
	VideoWriter outputcolor;
	VideoWriter outputdepth;
	//获取视频分辨率
	Size videoResolution = Size(640, 480);
	

	// 6. start
	VideoFrameRef  mColorFrame;
	VideoFrameRef  mDepthFrame;
	mDepthStream.start();
	mColorStream.start();
	int iMaxDepth = mDepthStream.getMaxPixelValue();
	int ImgNum = 0;
	char ImagesName[50];
	cv::Mat cImageBGR;
	cv::Mat mScaledDepth;
	cv::Mat mdepthimg;

	outputcolor.open(outputcolorPath, VideoWriter::fourcc('M', 'J', 'P', 'G'), 25.0, videoResolution, true);
	outputdepth.open(outputdepthPath, VideoWriter::fourcc('M', 'J', 'P', 'G'), 25.0, videoResolution, true);
	int count(300);
	while (count--)
	{
		// 7. check is color stream is available
		if (mColorStream.isValid())
		{
			// 7a. get color frame
			if (mColorStream.readFrame(&mColorFrame) == STATUS_OK)
			{
				// 7b. convert data to OpenCV format
				const cv::Mat mImageRGB(
					mColorFrame.getHeight(), mColorFrame.getWidth(),
					CV_8UC3, (void*)mColorFrame.getData());
				// 7c. convert form RGB to BGR

				cv::cvtColor(mImageRGB, cImageBGR, CV_RGB2BGR);
				outputcolor.write(cImageBGR); //把这一帧画面写入到outputVideo中
				// 7d. show image
				cv::imshow("Color Image", cImageBGR);
			}
		}

		// 8a. get depth frame
		if (mDepthStream.readFrame(&mDepthFrame) == STATUS_OK)
		{
			// 8b. convert data to OpenCV format
			const cv::Mat mImageDepth(
				mDepthFrame.getHeight(), mDepthFrame.getWidth(),
				CV_16UC1, (void*)mDepthFrame.getData());
			// 8c. re-map depth data [0,Max] to [0,255]
			mdepthimg = mImageDepth;
			
			mdepthimg = remove_over(mdepthimg);
			outputcolor << mdepthimg;
			//mdepthimg = RM_Gaussian_filter(mdepthimg);
			//cv::imwrite("depth_16.jpg", mImageDepth);
			
			mdepthimg.convertTo(mScaledDepth, CV_8U, 255.0 / iMaxDepth);
			// 8d. show image
			cv::imshow("Depth Image", mScaledDepth);
		}

		


		//pcl::PointCloud<pcl::PointXYZ>::Ptr output = image2PointCloud(cImageBGR, mdepthimg, camera);
		//cv::Mat show = paper_way(output, cImageBGR, camera);
		//cv::imshow("result", show);



			


		// //6a. check keyboard
		if (cv::waitKey(1) == 'q')
			break;
		//if (cv::waitKey(1) == 's')
		//{
		//	String filename_color =to_string(ImgNum) + ".jpg";
		//	String filename_depth = to_string(ImgNum) + ".png";
		//	String filename_plc = to_string(ImgNum) + ".pcd";
		//	
		//	imwrite(filename_color, cImageBGR);
		//	vector<int> compression_params;
		//	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
		//	compression_params.push_back(0); // 无压缩png.
		//	imwrite(filename_depth, mdepthimg, compression_params);
		//	cout << "size is" << output->size() << endl;
		//	pcl::io::savePCDFileBinary(filename_plc, *output);
		//	ImgNum++;
		//}
		//cloud_filter->clear();
		//view_point_cloud->clear();
		//output->clear();
	}

	// 9. stop
	mDepthStream.destroy();
	mColorStream.destroy();
	mDevice.close();
	OpenNI::shutdown();
	system("pause");
	return 0;
}




pcl::PointCloud<pcl::PointXYZ>::Ptr image2PointCloud(cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	int rowNumber = rgb.rows;
	int colNumber = rgb.cols;
	cloud->height = rowNumber;
	cloud->width = colNumber;
	cloud->points.resize(cloud->width * cloud->height);

	for (int m = 0; m < depth.rows; m++)
	{
		for (int n = 0; n < depth.cols; n++)
		{
			unsigned int num = m*colNumber + n;
			// 获取深度图中(m,n)处的值
			ushort d = depth.ptr<ushort>(m)[n];
			// d 可能没有值，若如此，跳过此点
			if (d == 0)
				continue;
			// d 存在值，则向点云增加一个点
			PointT p;

			// 计算这个点的空间坐标
			p.z = double(d) / camera.scale;
			p.x = (n - camera.cx) * p.z / camera.fx;
			p.y = (m - camera.cy) * p.z / camera.fy;

			// 从rgb图像中获取它的颜色
			// rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
			/*p.b = rgb.ptr<uchar>(m)[n * 3];
			p.g = rgb.ptr<uchar>(m)[n * 3 + 1];
			p.r = rgb.ptr<uchar>(m)[n * 3 + 2];*/
			// 把p加入到点云中
			cloud->points[num].x = p.x;
			cloud->points[num].y = p.y;
			cloud->points[num].z = p.z;
		}
	}
	// 设置并保存点云
	/*cloud->height = 1;
	cloud->width = cloud->points.size();*/
	cloud->is_dense = false;

	return cloud;
}


cv::Point3f point2dTo3d(cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera)
{
	cv::Point3f p; // 3D 点//是否需要标定等后期程序调试吧
	
	p.z = double(point.z) / camera.scale;
	p.x = (point.x - camera.cx) * p.z / camera.fx;
	p.y = (point.y - camera.cy) * p.z / camera.fy;
	return p;
}

cv::Point2f point3dTo2d(cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera)
{
	cv::Point3f p; // 3D 点//是否需要标定等后期程序调试吧
	cv::Point2f p_result = 0.0;

	p.z = double(point.z);
	p.x = point.x*camera.fx / p.z + camera.cx;
	p.y = point.y*camera.fy/p.z + camera.cy;
	p_result.x = p.x;
	p_result.y = p.y;
	return p_result;
}


cv::Mat remove_over(cv::Mat depth)
{
	for (int m = 0; m < depth.rows; m++)
	{
		for (int n = 0; n < depth.cols; n++)
		{
			// 获取深度图中(m,n)处的值
			ushort d = depth.ptr<ushort>(m)[n];
			// d 可能没有值，若如此，跳过此点
			if (d == 0)
				continue;
			else if (d > 1500)
			{
				depth.ptr<ushort>(m)[n] = 0;
			}

		}
	}
	return depth;
}


cv::Mat RM_Gaussian_filter(cv::Mat img_depth)
{
	double sum = 0;//求和
	double average_k = 0;//旧的均值
	double average_k1 = 0;//新补充的均值
	int median = 0;//中值
	double square_error = 0;//方差
	int k = 0;//对图像上的有效像素点进行计数
	int num_k = 0;
	for (int m = 0; m < img_depth.rows; m++)
	{
		for (int n = 0; n < img_depth.cols; n++)
		{
			// 获取深度图中(m,n)处的值
			ushort d = img_depth.ptr<ushort>(m)[n];
			// d 可能没有值，若如此，跳过此点
			if (d == 0)
				continue;
			// d 存在值，则计算一次方差和均值
			if (0 == k)
			{
				k++;
				sum = sum + d/10.00;
				square_error = 0;
				continue;
			}
			else
			{
				num_k = k; //记录上一时刻的点数
				average_k = sum / k; //计算上一时刻的均值
				sum = sum + d/10.00; //求现在点的距离和 
				average_k1 = sum / (++k); //计算这个时刻的均值
				 //计算这个时刻的方差
				square_error = square_error + num_k*(average_k - average_k1)*(average_k - average_k1) + (d/10.00 - average_k1)*(d/10.00 - average_k1);
			
				if ((img_depth.cols-1)==n)
				{
					double result = sqrt(square_error);
					if (result<2500)
					{
						for (int i = 0;i<=n;i++)
						{
							if (0 == img_depth.ptr<ushort>(m)[i])
							{
								img_depth.ptr<ushort>(m)[i] = average_k1*10;
							}
						}
					}
					else
					{
						for (int i = 0;i <= n;i++)
						{
							if (0 == img_depth.ptr<ushort>(m)[i]&&m>0&&m<479&&i>0&&i<639)
							{
								int left_up = img_depth.ptr<ushort>(m - 1)[i - 1];
								int up = img_depth.ptr<ushort>(m - 1)[i];
								int right_up = img_depth.ptr<ushort>(m - 1)[i + 1];
								int right = img_depth.ptr<ushort>(m)[i + 1];
								int right_down = img_depth.ptr<ushort>(m + 1)[i + 1];
								int down = img_depth.ptr<ushort>(m + 1)[i];
								int left_down = img_depth.ptr<ushort>(m + 1)[i - 1];
								int left = img_depth.ptr<ushort>(m)[i - 1];
								img_depth.ptr<ushort>(m)[i] = 1/12*(left_up+2*up+right_up+2*right+right_down+2*down+left_down+2*left);
							}
						}
					}
					sum = 0;//求和
					average_k = 0;//旧的均值
					average_k1 = 0;//新补充的均值
					median = 0;//中值
					square_error = 0;//方差
					k = 0;//对图像上的有效像素点进行计数
					num_k = 0;
				}
			}
		}
			
			
	}
	return img_depth;
}


cv::Mat paper_way(pcl::PointCloud<pcl::PointXYZ>::Ptr src,cv::Mat img, CAMERA_INTRINSIC_PARAMETERS& camera)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr result_vol(new pcl::PointCloud<pcl::PointXYZ>);

	//进行体素滤波
	pcl::VoxelGrid<pcl::PointXYZ> sor;
	sor.setInputCloud(src);
	sor.setLeafSize(0.03f, 0.03f, 0.03f);//体素大小设置为3*3*3cm
	sor.filter(*result_vol);


	//半径4cm进行搜索
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::RadiusOutlierRemoval<pcl::PointXYZ> result_radius;

	result_radius.setInputCloud(result_vol);
	result_radius.setRadiusSearch(0.04);
	result_radius.setMinNeighborsInRadius(2);
	// apply filter
	result_radius.filter(*cloud_filtered);



	//利用论文改进的RANSAC方法对地面点进行拟合
	//创建一个模型参数对象，用于记录结果
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	//inliers表示误差能容忍的点 记录的是点云的序号
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	// 创建一个分割器
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	// Optional，这个设置可以选定结果平面展示的点是分割掉的点还是分割剩下的点。
	seg.setOptimizeCoefficients(true);
	// Mandatory-设置目标几何形状
	seg.setModelType(pcl::SACMODEL_PLANE);
	//分割方法：随机采样法
	seg.setMethodType(pcl::SAC_RANSAC);
	//设置误差容忍范围，也就是我说过的阈值
	seg.setDistanceThreshold(0.02);
	//输入点云
	seg.setInputCloud(cloud_filtered);
	//分割点云
	seg.segment(*inliers, *coefficients);

	if (inliers->indices.size() == 0)
	{
		PCL_ERROR("Could not estimate a planar model for the given dataset.");
		return img;
	}
	std::cerr << "Model coefficients: " << coefficients->values[0] << " "
		<< coefficients->values[1] << " "
		<< coefficients->values[2] << " "
		<< coefficients->values[3] << std::endl;
	std::cerr << "Model inliers: " << inliers->indices.size() << std::endl;
	for (size_t i = 0; i < inliers->indices.size(); ++i)
	{
		double X = cloud_filtered->points[inliers->indices[i]].x;
		double Y = cloud_filtered->points[inliers->indices[i]].y;
		double Z = cloud_filtered->points[inliers->indices[i]].z;
		//把选中的内点重新投影到彩色图像中
		Point3f point_cl;
		point_cl.x = X;
		point_cl.y = Y;
		point_cl.z = Z;
		Point2f uv = point3dTo2d(point_cl, camera);
		img.at<Vec3b>(int(uv.y), int(uv.x))[0] = 0;//[0]蓝色通道
		img.at<Vec3b>(int(uv.y), int(uv.x))[1] = 0;//[1]绿色通道
		img.at<Vec3b>(int(uv.y), int(uv.x))[2] = 255;//[2]红色通道

	}
	return img;
}