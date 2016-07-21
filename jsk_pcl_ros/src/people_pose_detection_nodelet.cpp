// -*- mode: c++ -*-
/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2016, JSK Lab
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the JSK Lab nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

#define BOOST_PARAMETER_MAX_ARITY 7
#include "jsk_pcl_ros/people_pose_detection.h"
#include "jsk_recognition_utils/pcl_conversion_util.h"
#include "jsk_recognition_utils/pcl_util.h"

#include <pcl/people/ground_based_people_detection_app.h>

namespace jsk_pcl_ros {
  void PeoplePoseDetection::onInit() {
    DiagnosticNodelet::onInit();

    pnh_->param("gpu_device_index", gpu_device_index_, 0);
    pcl::gpu::setDevice(gpu_device_index_);
    pcl::gpu::printShortCudaDeviceInfo(gpu_device_index_);

    //selecting tree files
    std::vector<std::string> tree_files;
    tree_files.push_back("Data/forest1/tree_20.txt");
    tree_files.push_back("Data/forest2/tree_20.txt");
    tree_files.push_back("Data/forest3/tree_20.txt");
    tree_files.push_back("Data/forest4/tree_20.txt");
    int num_trees = (int)tree_files.size();

    rdf_ = boost::make_shared<RDFBodyPartsDetector::Ptr>(new RDFBodyPartsDetector(tree_files));

    ////////////////////////////////////////////////////////
    // dynamic reconfigure
    ////////////////////////////////////////////////////////
    srv_ = boost::make_shared<dynamic_reconfigure::Server<Config> >(*pnh_);
    dynamic_reconfigure::Server<Config>::CallbackType f =
      boost::bind(&PeoplePoseDetection::configCallback, this, _1, _2);
    srv_->setCallback(f);

    ////////////////////////////////////////////////////////
    // Publisher
    ////////////////////////////////////////////////////////
    // pub_cloud_ =
    //     advertise<sensor_msgs::PointCloud2>(*pnh_, "output", 1);

    onInitPostProcess();
  }

  void PeoplePoseDetection::configCallback(Config& config, uint32_t level) {
    boost::mutex::scoped_lock lock(mutex_);
    voxel_size_ = config.voxel_size;
    min_confidence_ = config.min_confidence;
    people_height_threshold_ = config.people_height_threshold;
    box_width_ = config.box_width;
    box_depth_ = config.box_depth;

  }

  void PeoplePoseDetection::subscribe() {
    sub_info_ =
      pnh_->subscribe("input/info", 1, &PeoplePoseDetection::infoCallback, this);
    sub_cloud_ = pnh_->subscribe("input", 1, &PeoplePoseDetection::detect, this);
  }

  void PeoplePoseDetection::unsubscribe() {
    sub_cloud_.shutdown();
    sub_info_.shutdown();
  }

  void PeoplePoseDetection::detect(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
      boost::mutex::scoped_lock lock(mutex_);

      pcl::PCLPointCloud2::Ptr input(new pcl::PCLPointCloud2);
      pcl_conversions::toPCL(*cloud_msg, *input);

      pcl::PointCloud<pcl::PointXYZRGBA> cloud_host;
      // pcl_conversions::toPCL(*cloud_msg, cloud_host);
      pcl::fromROSMsg(*cloud_msg, cloud_host);

      // process_return_ = people_detector_.process(input->makeShared());
      process_return_ = people_detector_.process(cloud_host.makeShared());

      const PeopleDetector::Labels& labels = people_detector_.rdf_detector_->getLabels();
      pcl::gpu::people::colorizeLabels(color_map_, labels, cmap_device_);

      int c;
      cmap_host_.width = cmap_device_.cols();
      cmap_host_.height = cmap_device_.rows();
      cmap_host_.points.resize(cmap_host_.width * cmap_host_.height);
      cmap_device_.download(cmap_host_.points, c);

      depth_host_.width = people_detector_.depth_device1_.cols();
      depth_host_.height = people_detector_.depth_device1_.rows();
      depth_host_.points.resize(depth_host_.width * depth_host_.height);
      people_detector_.depth_device1_.download(depth_host_.points, c);

  }

  // void PeoplePoseDetection::detect(
  //   const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
  //   boost::mutex::scoped_lock lock(mutex_);
  //   pcl::PointCloud<pcl::PointXYZRGBA>::Ptr input_cloud(
  //     new pcl::PointCloud<pcl::PointXYZRGBA>);
  //   pcl::fromROSMsg(*cloud_msg, *input_cloud);

  //   std::vector<pcl::people::PersonCluster<pcl::PointXYZRGBA> >
  //     clusters;  // vector containing persons clusters
  //   people_detector_.setInputCloud(input_cloud);
  //   people_detector_.setGround(ground_coeffs_);  // set floor coefficients
  //   people_detector_.compute(clusters);          // perform people detection

  //   jsk_recognition_msgs::BoundingBoxArray bounding_box_array;
  //   bounding_box_array.header = cloud_msg->header;
  //   jsk_recognition_msgs::BoundingBox bounding_box;
  //   bounding_box.header = cloud_msg->header;

  //   for (std::vector<pcl::people::PersonCluster<pcl::PointXYZRGBA> >::iterator it =
  //            clusters.begin();
  //        it != clusters.end(); ++it) {
  //     if (it->getPersonConfidence() > min_confidence_ &&
  //         it->getHeight() > people_height_threshold_) {
  //       bounding_box.pose.position.x = it->getCenter()[0];
  //       bounding_box.pose.position.y = it->getCenter()[1] + it->getHeight() / 2;
  //       bounding_box.pose.position.z = it->getCenter()[2];

  //       bounding_box.pose.orientation.x = 0.0;
  //       bounding_box.pose.orientation.y = 0.0;
  //       bounding_box.pose.orientation.z = 0.0;
  //       bounding_box.pose.orientation.w = 1.0;

  //       bounding_box.dimensions.x = box_width_;
  //       bounding_box.dimensions.y = it->getHeight() + 0.3;
  //       bounding_box.dimensions.z = box_depth_;

  //       bounding_box_array.boxes.push_back(bounding_box);
  //     }
  //   }
  //   pub_box_.publish(bounding_box_array);
  // }

  void PeoplePoseDetection::ground_coeffs_callback(
    const jsk_recognition_msgs::ModelCoefficientsArray::ConstPtr&
      coefficients_msg) {
    if (coefficients_msg->coefficients.size() >= 1) {
      set_ground_coeffs(coefficients_msg->coefficients[0]);
    }
  }

  void PeoplePoseDetection::set_ground_coeffs(
    const pcl_msgs::ModelCoefficients& coefficients) {
    boost::mutex::scoped_lock lock(mutex_);
    for (int i = 0; i < 4; ++i) {
      ground_coeffs_[i] = coefficients.values[i];
    }
  }

  void PeoplePoseDetection::infoCallback(
    const sensor_msgs::CameraInfo::ConstPtr& info_msg) {
    boost::mutex::scoped_lock lock(mutex_);
    latest_camera_info_ = info_msg;

    int cols = info_msg->height;
    int rows = info_msg->width;

    cmap_device_.create(rows, cols);
    cmap_host_.points.resize(cols * rows);
    depth_device_.create(rows, cols);
    image_device_.create(rows, cols);

    depth_host_.points.resize(cols * rows);

    rgba_host_.points.resize(cols * rows);
    rgb_host_.resize(cols * rows * 3);

    pcl::gpu::people::uploadColorMap(color_map_);
  }

  void PeoplePoseDetection::updateDiagnostic(
    diagnostic_updater::DiagnosticStatusWrapper& stat) {
    if (vital_checker_->isAlive()) {
      stat.summary(diagnostic_msgs::DiagnosticStatus::OK,
                   "PeoplePoseDetection running");
    } else {
      jsk_topic_tools::addDiagnosticErrorSummary("PeoplePoseDetection",
                                                 vital_checker_, stat);
    }
  }
}

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(jsk_pcl_ros::PeoplePoseDetection, nodelet::Nodelet);
