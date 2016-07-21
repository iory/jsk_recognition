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

#ifndef JSK_PCL_ROS_PEOPLE_POSE_DETECTION_H_
#define JSK_PCL_ROS_PEOPLE_POSE_DETECTION_H_

#include <jsk_topic_tools/diagnostic_nodelet.h>

#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <jsk_recognition_msgs/ModelCoefficientsArray.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>

#include <dynamic_reconfigure/server.h>
#include "jsk_pcl_ros/PeoplePoseDetectionConfig.h"

#include <pcl/people/ground_based_people_detection_app.h>
#include <pcl/gpu/containers/initialization.h>
#include <pcl/gpu/people/people_detector.h>
#include <pcl/gpu/people/colormap.h>

namespace jsk_pcl_ros {
  class PeoplePoseDetection : public jsk_topic_tools::DiagnosticNodelet {
   public:
    PeoplePoseDetection() : DiagnosticNodelet("PeoplePoseDetection") {}

    typedef pcl::gpu::people::PeopleDetector PeopleDetector;
    typedef jsk_pcl_ros::PeoplePoseDetectionConfig Config;
    typedef pcl::gpu::people::RDFBodyPartsDetector RDFBodyPartsDetector;

   protected:
    ////////////////////////////////////////////////////////
    // methods
    ////////////////////////////////////////////////////////
    virtual void onInit();

    virtual void ground_coeffs_callback(
      const jsk_recognition_msgs::ModelCoefficientsArray::ConstPtr&
        coefficients_msg);
    virtual void set_ground_coeffs(
      const pcl_msgs::ModelCoefficients& coefficients);
    virtual void infoCallback(
      const sensor_msgs::CameraInfo::ConstPtr& info_msg);
    virtual void subscribe();
    virtual void unsubscribe();
    virtual void updateDiagnostic(
      diagnostic_updater::DiagnosticStatusWrapper& stat);
    virtual void detect(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg);

    ////////////////////////////////////////////////////////
    // ROS variables
    ////////////////////////////////////////////////////////
    ros::Subscriber sub_cloud_;
    ros::Subscriber sub_info_;
    ros::Subscriber sub_coefficients_;

    ros::Publisher pub_box_;

    ////////////////////////////////////////////////////////
    // dynamic reconfigure
    ////////////////////////////////////////////////////////
    boost::shared_ptr<dynamic_reconfigure::Server<Config> > srv_;
    void configCallback(Config& config, uint32_t level);

    ////////////////////////////////////////////////////////
    // parameters
    ////////////////////////////////////////////////////////
    boost::mutex mutex_;

    sensor_msgs::CameraInfo::ConstPtr latest_camera_info_;

    pcl::people::PersonClassifier<pcl::RGB> person_classifier_;

    Eigen::VectorXf ground_coeffs_;

    int gpu_device_index_;
    double box_depth_;
    double box_width_;
    double min_confidence_;
    double people_height_threshold_;
    double voxel_size_;
    int queue_size_;
    std::string trained_filename_;

    boost::mutex data_ready_mutex_;
    boost::condition_variable data_ready_cond_;
    bool cloud_cb_;
    bool write_;
    bool exit_;
    int time_ms_;
    int counter_;
    int process_return_;
    PeopleDetector people_detector_;
    PeopleDetector::Image cmap_device_;
    pcl::PointCloud<pcl::RGB> cmap_host_;

    pcl::gpu::people::PeopleDetector::Depth depth_device_;
    pcl::gpu::people::PeopleDetector::Image image_device_;
    pcl::PointCloud<unsigned short> depth_host_;
    pcl::PointCloud<pcl::RGB> rgba_host_;
    std::vector<unsigned char> rgb_host_;

    pcl::PointCloud<pcl::PointXYZRGBA> cloud_host_;
    pcl::gpu::DeviceArray<pcl::RGB> color_map_;

    boost::shared_ptr<RDFBodyPartsDetector::Ptr> rdf_;

   private:
  };
}

#endif
