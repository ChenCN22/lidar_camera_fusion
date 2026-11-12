/**
 * lidar_camera_fusion_node.cpp
 *
 * Registers one or two cameras against a single LiDAR.  For each camera it
 * subscribes to that camera's image topic, reads that camera's extrinsic
 * calibration json, and publishes:
 *   1. A colored PointCloud2 (each point painted with its camera-image colour)
 *      at the LiDAR rate (~10 Hz).
 *   2. An overlap image (LiDAR points projected and drawn on top of the camera
 *      image) at the image rate (~30 Hz / FPS).
 *
 * The second camera is optional: leave "calib_path_back" empty and the node
 * runs with one camera.  The two cameras are entirely
 * independent copies of the same pipeline - separate calibration, separate
 * cloud FIFO, separate output topics.  They share only the LiDAR subscription,
 * the depth colormap and the tuning parameters.
 *
 * Because the images of the rig this was written for lag the LiDAR clouds by
 * ~300 ms, a cloud FIFO queue of configurable depth (default 3 frames) is
 * used: the oldest cloud in the queue is colourised with the current image,
 * so the depth-3 delay at 10 Hz compensates for the 300 ms offset.  Each
 * camera owns its own FIFO - a shared one would not work, because the image
 * callback pops from it and the two cameras would steal each other's clouds.
 *
 * Direct subscriptions with manual queue management drive both paths; there is
 * no message_filters synchronizer.
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <std_msgs/msg/header.hpp>

#ifdef CV_BRIDGE_INCLUDE_H
#include <cv_bridge/cv_bridge.h>
#else
#include <cv_bridge/cv_bridge.hpp>
#endif

#include <opencv2/opencv.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <utility>
#include <memory>
#include <vector>

// ---------------------------------------------------------------------------
// Camera projection helpers
// ---------------------------------------------------------------------------

namespace {

/**
 * @brief Project a 3-D point (in camera frame) to a 2-D pixel using the
 *        plumb_bob (standard radial-tangential) distortion model.
 *        Matches the PinholeProjection in vlcal exactly.
 */
Eigen::Vector2d projectPinhole(
  const Eigen::Vector3d& p,
  const std::vector<double>& intrinsics,
  const std::vector<double>& dist)
{
  const double xn = p.x() / p.z();
  const double yn = p.y() / p.z();

  const double k1 = dist.size() > 0 ? dist[0] : 0.0;
  const double k2 = dist.size() > 1 ? dist[1] : 0.0;
  const double p1 = dist.size() > 2 ? dist[2] : 0.0;
  const double p2 = dist.size() > 3 ? dist[3] : 0.0;
  const double k3 = dist.size() > 4 ? dist[4] : 0.0;

  const double r2 = xn * xn + yn * yn;
  const double r4 = r2 * r2;
  const double r6 = r2 * r4;

  const double rc = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
  const double xd = rc * xn + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn);
  const double yd = rc * yn + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn;

  return {intrinsics[0] * xd + intrinsics[2], intrinsics[1] * yd + intrinsics[3]};
}

/**
 * @brief Project using the fisheye (equidistant) distortion model.
 *        Matches the FisheyeProjection in vlcal exactly.
 */
Eigen::Vector2d projectFisheye(
  const Eigen::Vector3d& p,
  const std::vector<double>& intrinsics,
  const std::vector<double>& dist)
{
  const double r = p.head<2>().norm();
  const double theta = std::atan2(r, std::abs(p.z()));
  const double t2 = theta * theta;
  const double t4 = t2 * t2;
  const double t6 = t2 * t4;
  const double t8 = t4 * t4;

  const double k1 = dist.size() > 0 ? dist[0] : 0.0;
  const double k2 = dist.size() > 1 ? dist[1] : 0.0;
  const double k3 = dist.size() > 2 ? dist[2] : 0.0;
  const double k4 = dist.size() > 3 ? dist[3] : 0.0;

  const double theta_d = theta * (1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8);
  const double scale = (r > 1e-9) ? theta_d / r : 1.0;

  return {
    intrinsics[0] * scale * p.x() + intrinsics[2],
    intrinsics[1] * scale * p.y() + intrinsics[3]};
}

/**
 * @brief Project using the equirectangular (360°) model.
 *        Matches the EquirectangularProjection in vlcal exactly.
 *        intrinsics: [width, height]
 */
Eigen::Vector2d projectEquirectangular(
  const Eigen::Vector3d& p,
  const std::vector<double>& intrinsics)
{
  if (p.squaredNorm() < 1e-6) {
    return {intrinsics[0] / 2.0, intrinsics[1] / 2.0};
  }
  const Eigen::Vector3d b = p.normalized();
  const double lat = -std::asin(b.y());
  const double lon = std::atan2(b.x(), b.z());
  return {
    intrinsics[0] * (0.5 + lon / (2.0 * M_PI)),
    intrinsics[1] * (0.5 - lat / M_PI)};
}

/**
 * @brief Project using the unified omnidirectional (mirror) camera model with
 *        plumb_bob distortion.  Matches
 *        camera::OmnidirectionalProjection in direct_visual_lidar_calibration.
 *        intrinsics: [fx, fy, cx, cy, xi]
 *        distortion: [k1, k2, p1, p2]
 *
 *        Returns {NaN, NaN} if the point falls behind the unit sphere + xi
 *        offset (i.e. outside the camera's valid projection hemisphere).
 */
Eigen::Vector2d projectOmnidir(
  const Eigen::Vector3d& p,
  const std::vector<double>& intrinsics,
  const std::vector<double>& dist)
{
  const double fx = intrinsics[0];
  const double fy = intrinsics[1];
  const double cx = intrinsics[2];
  const double cy = intrinsics[3];
  const double xi = intrinsics.size() >= 5 ? intrinsics[4] : 1.0;

  const double k1 = dist.size() > 0 ? dist[0] : 0.0;
  const double k2 = dist.size() > 1 ? dist[1] : 0.0;
  const double p1 = dist.size() > 2 ? dist[2] : 0.0;
  const double p2 = dist.size() > 3 ? dist[3] : 0.0;

  // Unified sphere → shifted pinhole: pt_u = pt_s.xy / (pt_s.z + xi)
  const Eigen::Vector3d pt_s = p.normalized();
  const double denom = pt_s.z() + xi;
  if (denom < 1e-6) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {nan, nan};
  }
  const Eigen::Vector2d pt_u = pt_s.head<2>() / denom;

  const double r2 = pt_u.squaredNorm();
  const double r4 = r2 * r2;

  const double dr = 1.0 + k1 * r2 + k2 * r4;
  const double x2 = pt_u[0] * pt_u[0];
  const double y2 = pt_u[1] * pt_u[1];
  const double xy = pt_u[0] * pt_u[1];

  const double nx = pt_u[0] * dr + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2);
  const double ny = pt_u[1] * dr + p1 * (r2 + 2.0 * y2) + 2.0 * p2 * xy;

  return {fx * nx + cx, fy * ny + cy};
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

class LidarCameraFusionNode : public rclcpp::Node
{
public:
  using ImageMsg    = sensor_msgs::msg::Image;
  using CloudMsg    = sensor_msgs::msg::PointCloud2;
  using PointField  = sensor_msgs::msg::PointField;

  LidarCameraFusionNode()
  : Node("lidar_camera_fusion")
  {
    // -----------------------------------------------------------------------
    // Parameters
    // -----------------------------------------------------------------------
    declare_parameter("image_topic",          "/image");
    declare_parameter("cloud_topic",          "/ouster/points");
    declare_parameter("calib_path",           "/tmp/preprocessed");
    declare_parameter("queue_size",           3);
    // Image subscription QoS: "reliable" (default) matches a RELIABLE image
    // publisher so lost UDP fragments are repaired instead of dropping the
    // frame; "sensor" = BEST_EFFORT SensorDataQoS.
    declare_parameter("image_qos",            "reliable");
    declare_parameter("overlap_image_topic",  "/overlap_image");
    declare_parameter("colored_cloud_topic",  "/colored_cloud");
    declare_parameter("point_radius",         2);
    // Drop returns further than this from the sensor, in metres.  <= 0 disables
    // the filter and keeps the whole cloud.
    declare_parameter("max_distance",         8.0);
    // Second camera.  Leave "calib_path_back" empty to run with one lens.
    declare_parameter("image_topic_back",         "/camera_back");
    declare_parameter("calib_path_back",          "");
    declare_parameter("overlap_image_topic_back", "/overlap_image_back");
    declare_parameter("colored_cloud_topic_back", "/colored_cloud_back");

    const auto cloud_topic   = get_parameter("cloud_topic").as_string();
    queue_size_              = get_parameter("queue_size").as_int();
    point_radius_            = get_parameter("point_radius").as_int();
    max_distance_            = get_parameter("max_distance").as_double();

    RCLCPP_INFO(get_logger(), "cloud_topic  : %s", cloud_topic.c_str());
    RCLCPP_INFO(get_logger(), "queue_size   : %d", queue_size_);
    if (max_distance_ > 0.0) {
      RCLCPP_INFO(get_logger(), "max_distance : %.2f m", max_distance_);
    } else {
      RCLCPP_INFO(get_logger(), "max_distance : disabled (keeping all points)");
    }

    // -----------------------------------------------------------------------
    // Cameras
    //
    // Each camera is an independent copy of the same registration pipeline:
    // its own calibration, its own cloud FIFO, its own overlay image and its
    // own coloured cloud.  The LiDAR, the colormap and the tuning parameters
    // are the only things they share.
    // -----------------------------------------------------------------------
    cameras_.reserve(2);

    Camera front;
    front.index               = 0;
    front.name                = "front";
    front.image_topic         = get_parameter("image_topic").as_string();
    front.calib_path          = get_parameter("calib_path").as_string();
    front.overlay_topic       = get_parameter("overlap_image_topic").as_string();
    front.colored_cloud_topic = get_parameter("colored_cloud_topic").as_string();
    cameras_.push_back(std::move(front));

    const auto back_calib = get_parameter("calib_path_back").as_string();
    if (back_calib.empty()) {
      RCLCPP_INFO(get_logger(),
        "calib_path_back is empty - running with one camera");
    } else {
      Camera back;
      back.index               = 1;
      back.name                = "back";
      back.image_topic         = get_parameter("image_topic_back").as_string();
      back.calib_path          = back_calib;
      back.overlay_topic       = get_parameter("overlap_image_topic_back").as_string();
      back.colored_cloud_topic = get_parameter("colored_cloud_topic_back").as_string();
      cameras_.push_back(std::move(back));
    }

    // -----------------------------------------------------------------------
    // Calibration + publishers, per camera
    // -----------------------------------------------------------------------
    for (auto& cam : cameras_) {
      RCLCPP_INFO(get_logger(), "camera '%s' : %s  <-  %s",
        cam.name.c_str(), cam.image_topic.c_str(), cam.calib_path.c_str());
      loadCalib(cam);
      cam.overlap_image_pub = create_publisher<ImageMsg>(cam.overlay_topic, 10);
      cam.colored_cloud_pub = create_publisher<CloudMsg>(cam.colored_cloud_topic, 10);
    }

    // -----------------------------------------------------------------------
    // Direct subscriptions with manual queue management
    //
    // The image topic lags the cloud topic by ~300 ms. We use separate
    // direct subscriptions with a FIFO queue: as clouds arrive, they're
    // queued; as images arrive, if the queue has accumulated queue_size_
    // frames (~300 ms), we pop the oldest cloud and colorize it with the
    // current image.
    //
    // The cloud uses SensorDataQoS (BEST_EFFORT, volatile); the image QoS is
    // selected by the image_qos parameter.
    // -----------------------------------------------------------------------
    // Subscriptions are created last and capture the camera INDEX, so nothing
    // can dangle if cameras_ ever reallocates.
    const std::string image_qos_name = get_parameter("image_qos").as_string();
    const rclcpp::QoS image_qos = (image_qos_name == "sensor")
      ? rclcpp::QoS(rclcpp::SensorDataQoS())
      : rclcpp::QoS(rclcpp::KeepLast(3)).reliable().durability_volatile();
    RCLCPP_INFO(get_logger(), "image subscription QoS: %s", image_qos_name.c_str());
    for (size_t i = 0; i < cameras_.size(); ++i) {
      cameras_[i].image_sub = create_subscription<ImageMsg>(
        cameras_[i].image_topic,
        image_qos,
        [this, i](const ImageMsg::ConstSharedPtr msg) { imageCallback(i, msg); });
    }

    cloud_direct_sub_ = create_subscription<CloudMsg>(
      cloud_topic,
      rclcpp::SensorDataQoS(),
      std::bind(&LidarCameraFusionNode::cloudCallback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "lidar_camera_fusion node ready (%zu camera%s)",
      cameras_.size(), cameras_.size() == 1 ? "" : "s");
  }

private:
  /// Everything that belongs to one lens.  The LiDAR, the colormap and the
  /// tuning parameters (queue_size / point_radius / max_distance) are shared;
  /// this is the state that is not.
  struct Camera
  {
    size_t               index{0};              ///< position in cameras_
    std::string          name;                  ///< "front" / "back", for logs
    std::string          image_topic;
    std::string          calib_path;
    std::string          overlay_topic;
    std::string          colored_cloud_topic;

    // --- calibration (written by loadCalib) ---
    Eigen::Isometry3d    T_camera_lidar{Eigen::Isometry3d::Identity()};
    // This lens' optical axis (+z) expressed in the LiDAR frame.  Used to
    // decide, for a point both lenses can see, which lens owns it - see the
    // ownership test in colorizeAndPublish.
    Eigen::Vector3d      optical_axis_lidar{Eigen::Vector3d::UnitX()};
    std::string          camera_model;
    std::vector<double>  intrinsics;
    std::vector<double>  distortion;

    // --- io ---
    rclcpp::Subscription<ImageMsg>::SharedPtr  image_sub;
    rclcpp::Publisher<ImageMsg>::SharedPtr     overlap_image_pub;
    rclcpp::Publisher<CloudMsg>::SharedPtr     colored_cloud_pub;

    // --- state (guarded by state_mutex_) ---
    std::deque<CloudMsg::ConstSharedPtr>            cloud_deque;
    std::vector<std::pair<cv::Point2i, cv::Vec3b>>  colored_points_2d;
  };

  // =========================================================================
  // Calibration loading
  // =========================================================================

  void loadCalib(Camera& cam)
  {
    std::ifstream ifs(cam.calib_path);
    if (!ifs) {
      RCLCPP_FATAL(get_logger(), "Cannot open %s", cam.calib_path.c_str());
      throw std::runtime_error("Cannot open calib.json");
    }

    nlohmann::json cfg;
    ifs >> cfg;

    cam.camera_model = cfg["camera"]["camera_model"].get<std::string>();
    cam.intrinsics   = cfg["camera"]["intrinsics"].get<std::vector<double>>();
    cam.distortion   = cfg["camera"]["distortion_coeffs"].get<std::vector<double>>();

    RCLCPP_INFO(get_logger(), "  camera_model : %s", cam.camera_model.c_str());

    // Prefer the final calibrated transform; fall back to the automatic guess.
    std::vector<double> tum;
    if (cfg.count("results") && cfg["results"].count("T_lidar_camera")) {
      RCLCPP_INFO(get_logger(), "  using T_lidar_camera (calibrated result)");
      tum = cfg["results"]["T_lidar_camera"].get<std::vector<double>>();
    } else if (cfg.count("results") && cfg["results"].count("init_T_lidar_camera_auto")) {
      RCLCPP_INFO(get_logger(), "  using init_T_lidar_camera_auto");
      tum = cfg["results"]["init_T_lidar_camera_auto"].get<std::vector<double>>();
    } else {
      RCLCPP_FATAL(get_logger(), "No T_lidar_camera in calib.json results");
      throw std::runtime_error("No calibration transform found in calib.json");
    }

    // TUM format: tx ty tz  qx qy qz qw
    Eigen::Isometry3d T_lidar_camera = Eigen::Isometry3d::Identity();
    T_lidar_camera.translation() << tum[0], tum[1], tum[2];
    T_lidar_camera.linear() =
      Eigen::Quaterniond(tum[6], tum[3], tum[4], tum[5]).normalized().toRotationMatrix();

    cam.T_camera_lidar = T_lidar_camera.inverse();

    // Optical axis (camera +z) in the LiDAR frame = R_lidar_camera * e_z.
    // R_lidar_camera is the rotation of T_lidar_camera, i.e. the transpose of
    // R_camera_lidar.  This is a unit vector already (rotation column).
    cam.optical_axis_lidar =
      cam.T_camera_lidar.linear().transpose().col(2).normalized();

    RCLCPP_INFO_STREAM(get_logger(),
      "  T_camera_lidar:\n" << cam.T_camera_lidar.matrix());
    RCLCPP_INFO(get_logger(), "  optical axis in LiDAR frame: [%.3f %.3f %.3f]",
      cam.optical_axis_lidar.x(), cam.optical_axis_lidar.y(),
      cam.optical_axis_lidar.z());
  }

  // =========================================================================
  // Camera projection
  // =========================================================================

  /**
   * @brief Project a point in camera-frame to pixel coordinates.
   * @return Pixel (u, v).  May be out-of-image – caller must bounds-check.
   */
  Eigen::Vector2d projectPoint(const Camera& cam, const Eigen::Vector3d& pt_cam) const
  {
    if (cam.camera_model == "fisheye") {
      return projectFisheye(pt_cam, cam.intrinsics, cam.distortion);
    }
    if (cam.camera_model == "equirectangular") {
      return projectEquirectangular(pt_cam, cam.intrinsics);
    }
    if (cam.camera_model == "omnidir") {
      return projectOmnidir(pt_cam, cam.intrinsics, cam.distortion);
    }
    // plumb_bob / atan: use pinhole-based projection
    return projectPinhole(pt_cam, cam.intrinsics, cam.distortion);
  }

  // =========================================================================
  // Cloud callback – 10 Hz (receives point clouds)
  // =========================================================================

  void cloudCallback(const CloudMsg::ConstSharedPtr& msg)
  {
    std::lock_guard<std::mutex> lk(state_mutex_);

    // Every camera gets its OWN FIFO holding the same message.  A shared queue
    // would not work: imageCallback pops from it, so the two cameras would
    // steal each other's clouds and each would only see about half of them.
    // Pushing the shared_ptr into both copies a refcount, not the payload.
    for (auto& cam : cameras_) {
      cam.cloud_deque.push_back(msg);

      // Keep only the most recent queue_size_ frames in the FIFO.
      // The oldest frame in the queue is ~(queue_size * 100ms) = 300ms old,
      // which compensates for the image lag.
      while (static_cast<int>(cam.cloud_deque.size()) > queue_size_) {
        cam.cloud_deque.pop_front();
      }
    }
  }

  // =========================================================================
  // Image callback – 30 Hz: trigger colorisation & overlay publishing
  // =========================================================================

  void imageCallback(size_t idx, const ImageMsg::ConstSharedPtr& msg)
  {
    Camera& cam = cameras_[idx];

    // Convert to BGR
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "cv_bridge (%s): %s", cam.name.c_str(), e.what());
      return;
    }

    // No clone here: colorize only READS the image, and the overlay is drawn
    // into a freshly allocated output message below (one copy per frame).
    const cv::Mat& img = cv_ptr->image;

    // If the cloud queue is full, pop the oldest cloud and colorize.
    CloudMsg::ConstSharedPtr oldest_cloud;
    {
      std::lock_guard<std::mutex> lk(state_mutex_);
      if (static_cast<int>(cam.cloud_deque.size()) >= queue_size_) {
        oldest_cloud = cam.cloud_deque.front();
        cam.cloud_deque.pop_front();
      }
    }

    if (oldest_cloud) {
      colorizeAndPublish(cam, oldest_cloud, img);
    }

    // Draw the most recent colorisation pass onto the current frame.
    std::vector<std::pair<cv::Point2i, cv::Vec3b>> pts;
    {
      std::lock_guard<std::mutex> lk(state_mutex_);
      pts = cam.colored_points_2d;
    }

    // Allocate the outgoing image and draw straight into its data buffer -
    // one copy (img -> overlay) instead of three.
    auto out_img = std::make_unique<ImageMsg>();
    out_img->header       = msg->header;
    out_img->height       = static_cast<uint32_t>(img.rows);
    out_img->width        = static_cast<uint32_t>(img.cols);
    out_img->encoding     = "bgr8";
    out_img->is_bigendian = 0;
    out_img->step         = static_cast<uint32_t>(img.cols) * 3;
    out_img->data.resize(static_cast<size_t>(out_img->step) * img.rows);

    cv::Mat overlay(img.rows, img.cols, CV_8UC3,
                    out_img->data.data(), out_img->step);
    img.copyTo(overlay);
    for (const auto& [pt, bgr] : pts) {
      if (pt.x >= 0 && pt.x < overlay.cols && pt.y >= 0 && pt.y < overlay.rows) {
        cv::circle(overlay, pt, point_radius_,
          cv::Scalar(bgr[0], bgr[1], bgr[2]), cv::FILLED);
      }
    }

    cam.overlap_image_pub->publish(std::move(out_img));
  }

  // =========================================================================
  // Colorisation + publish
  // =========================================================================

  /**
   * @brief Colorize cloud points: RGB for colored_cloud output, depth-based
   *        colormap for the overlay image.
   */
  void colorizeAndPublish(
    Camera& cam,
    const CloudMsg::ConstSharedPtr& cloud_msg,
    const cv::Mat& image)
  {
    // ------------------------------------------------------------------
    // Parse the input PointCloud2 field offsets and data types
    // ------------------------------------------------------------------
    int x_off = -1, y_off = -1, z_off = -1;
    uint8_t xyz_type = PointField::FLOAT32;

    for (const auto& f : cloud_msg->fields) {
      if      (f.name == "x") { x_off = static_cast<int>(f.offset); xyz_type = f.datatype; }
      else if (f.name == "y") { y_off = static_cast<int>(f.offset); }
      else if (f.name == "z") { z_off = static_cast<int>(f.offset); }
    }

    if (x_off < 0 || y_off < 0 || z_off < 0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "[%s] input cloud is missing x/y/z fields – skipping", cam.name.c_str());
      return;
    }

    const int   num_pts    = static_cast<int>(cloud_msg->width * cloud_msg->height);
    const int   src_step   = static_cast<int>(cloud_msg->point_step);
    const bool  is_float64 = (xyz_type == PointField::FLOAT64);

    const uint8_t* src_base = cloud_msg->data.data();

    // ------------------------------------------------------------------
    // Single pass over the raw cloud: distance-filter, transform, project.
    //
    // Only points that actually land in THIS lens' image are kept.  Points in
    // the opposite hemisphere never project into bounds, so they are dropped
    // here instead of being carried in the output as black filler.
    // ------------------------------------------------------------------
    struct Hit { float x, y, z; uint32_t rgb; int32_t u, v; float depth; };
    static thread_local std::vector<Hit> hits;   // reused across frames
    hits.clear();
    hits.reserve(static_cast<size_t>(num_pts) / 2);

    const bool is_wide = (cam.camera_model == "equirectangular" ||
                          cam.camera_model == "omnidir" ||
                          cam.camera_model == "fisheye");

    double min_depth = std::numeric_limits<double>::max();
    double max_depth = std::numeric_limits<double>::lowest();

    for (int i = 0; i < num_pts; ++i) {
      const uint8_t* src = src_base + i * src_step;

      float x, y, z;
      if (is_float64) {
        x = static_cast<float>(*reinterpret_cast<const double*>(src + x_off));
        y = static_cast<float>(*reinterpret_cast<const double*>(src + y_off));
        z = static_cast<float>(*reinterpret_cast<const double*>(src + z_off));
      } else {
        x = *reinterpret_cast<const float*>(src + x_off);
        y = *reinterpret_cast<const float*>(src + y_off);
        z = *reinterpret_cast<const float*>(src + z_off);
      }

      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        continue;
      }

      // Distance filter: range from the LiDAR origin, which is what "N metres
      // away" means physically.
      const double range = std::sqrt(static_cast<double>(x) * x +
                                     static_cast<double>(y) * y +
                                     static_cast<double>(z) * z);
      if (max_distance_ > 0.0 && range > max_distance_) {
        continue;
      }

      // Ownership: where two lenses overlap, the same physical point would
      // otherwise appear in BOTH clouds - coloured from opposite fisheye edges
      // and updated at different times - which flickers when they are rendered
      // together.  Assign each point to the single lens whose optical axis
      // points most directly at it (largest dot product with the view
      // direction), so the two clouds are a clean, disjoint partition.  Ties go
      // to the lower-index lens.  This uses the direction from the LiDAR origin,
      // which ignores the baseline between the lenses.
      if (cameras_.size() > 1) {
        const Eigen::Vector3d dir(static_cast<double>(x) / range,
                                  static_cast<double>(y) / range,
                                  static_cast<double>(z) / range);
        const double dot_self = dir.dot(cam.optical_axis_lidar);
        bool owned = true;
        for (const auto& other : cameras_) {
          if (other.index == cam.index) {
            continue;
          }
          const double dot_other = dir.dot(other.optical_axis_lidar);
          if (dot_other > dot_self ||
              (dot_other == dot_self && other.index < cam.index)) {
            owned = false;
            break;
          }
        }
        if (!owned) {
          continue;   // the other lens owns this point
        }
      }

      // Transform to camera frame.
      const Eigen::Vector4d pt_lidar(x, y, z, 1.0);
      const Eigen::Vector4d pt_cam = cam.T_camera_lidar * pt_lidar;

      // Reject points clearly behind a pinhole-like camera.  Wide-angle models
      // legitimately see z <= 0; projectOmnidir returns NaN outside its valid
      // hemisphere.
      if (!is_wide && pt_cam.z() <= 0.01) {
        continue;
      }

      const Eigen::Vector2d px = projectPoint(cam, pt_cam.head<3>());
      if (!std::isfinite(px.x()) || !std::isfinite(px.y())) {
        continue;
      }
      const int u = static_cast<int>(std::round(px.x()));
      const int v = static_cast<int>(std::round(px.y()));
      if (u < 0 || u >= image.cols || v < 0 || v >= image.rows) {
        continue;   // outside this lens' image -> other hemisphere, drop it
      }

      // Radial distance rather than z, so omnidir points with a valid but
      // non-positive z still get a monotone colormap scaling.
      const double depth = pt_cam.head<3>().norm();
      const cv::Vec3b bgr = image.at<cv::Vec3b>(v, u);
      const uint32_t rgb32 = (static_cast<uint32_t>(bgr[2]) << 16) |
                             (static_cast<uint32_t>(bgr[1]) <<  8) |
                              static_cast<uint32_t>(bgr[0]);

      hits.push_back({x, y, z, rgb32, u, v, static_cast<float>(depth)});
      min_depth = std::min(min_depth, depth);
      max_depth = std::max(max_depth, depth);
    }

    const double depth_range = (max_depth > min_depth) ? (max_depth - min_depth) : 1.0;

    // ------------------------------------------------------------------
    // Build the cropped coloured cloud: x y z (FLOAT32) + rgb, one entry per
    // kept point.  Because every point here is a valid coloured return the
    // cloud is dense.
    // ------------------------------------------------------------------
    const size_t n_out = hits.size();

    auto out = std::make_unique<CloudMsg>();
    out->header       = cloud_msg->header;
    out->height       = 1;
    out->width        = static_cast<uint32_t>(n_out);
    out->is_dense     = true;
    out->is_bigendian = false;

    const auto push_field = [&](const std::string& name, uint32_t offset, uint8_t dtype) {
      PointField f;
      f.name     = name;
      f.offset   = offset;
      f.datatype = dtype;
      f.count    = 1;
      out->fields.push_back(f);
    };
    push_field("x",   0,  PointField::FLOAT32);
    push_field("y",   4,  PointField::FLOAT32);
    push_field("z",   8,  PointField::FLOAT32);
    push_field("rgb", 12, PointField::FLOAT32);  // PCL-convention packed RGB
    out->point_step = 16;
    out->row_step   = out->point_step * static_cast<uint32_t>(n_out);
    out->data.resize(out->row_step);

    // Overlay projections (depth-colormapped) for imageCallback to draw.
    std::vector<std::pair<cv::Point2i, cv::Vec3b>> new_pts_2d_overlay;
    new_pts_2d_overlay.reserve(n_out);

    uint8_t* dst = out->data.data();
    for (const Hit& h : hits) {
      *reinterpret_cast<float*>(dst + 0) = h.x;
      *reinterpret_cast<float*>(dst + 4) = h.y;
      *reinterpret_cast<float*>(dst + 8) = h.z;
      std::memcpy(dst + 12, &h.rgb, 4);
      dst += out->point_step;

      const double depth_norm = (h.depth - min_depth) / depth_range;  // [0, 1]
      const int colormap_idx =
        std::clamp(static_cast<int>(depth_norm * 255.0), 0, 255);
      new_pts_2d_overlay.emplace_back(cv::Point2i(h.u, h.v),
                                      depth_colormap_[colormap_idx]);
    }

    // Hand the overlay projections to imageCallback.
    {
      std::lock_guard<std::mutex> lk(state_mutex_);
      cam.colored_points_2d = std::move(new_pts_2d_overlay);
    }

    RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 5000,
      "[%s] %zu points coloured within %.2f m",
      cam.name.c_str(), n_out, max_distance_);

    cam.colored_cloud_pub->publish(std::move(out));
  }

private:
  // Precomputed Turbo colormap (256 colors)
  static constexpr uint8_t depth_colormap_data_[256][3] = {
    {48, 18, 59}, {50, 21, 67}, {51, 24, 74}, {52, 27, 81}, {53, 30, 88}, {54, 33, 95}, {55, 36, 102}, {56, 39, 109},
    {57, 42, 115}, {58, 45, 121}, {59, 47, 128}, {60, 50, 134}, {61, 53, 139}, {62, 56, 145}, {63, 59, 151}, {63, 62, 156},
    {64, 64, 162}, {65, 67, 167}, {65, 70, 172}, {66, 73, 177}, {66, 75, 181}, {67, 78, 186}, {68, 81, 191}, {68, 84, 195},
    {68, 86, 199}, {69, 89, 203}, {69, 92, 206}, {69, 94, 210}, {70, 97, 213}, {70, 100, 216}, {70, 102, 219}, {70, 105, 221},
    {71, 107, 224}, {71, 110, 226}, {71, 113, 228}, {71, 115, 230}, {71, 118, 232}, {71, 120, 233}, {71, 123, 235}, {70, 125, 236},
    {70, 128, 237}, {70, 130, 238}, {70, 133, 239}, {69, 135, 240}, {69, 138, 240}, {68, 140, 241}, {67, 143, 241}, {66, 145, 241},
    {65, 148, 241}, {64, 150, 241}, {62, 153, 240}, {61, 155, 240}, {59, 158, 239}, {58, 160, 239}, {56, 163, 238}, {54, 165, 237},
    {52, 168, 236}, {50, 170, 235}, {48, 173, 234}, {46, 175, 233}, {44, 178, 231}, {42, 180, 230}, {40, 183, 228}, {37, 185, 226},
    {35, 188, 224}, {32, 190, 222}, {30, 193, 220}, {27, 195, 218}, {24, 198, 215}, {22, 200, 213}, {19, 203, 210}, {16, 205, 207},
    {14, 208, 204}, {11, 210, 201}, {8, 213, 198}, {6, 215, 195}, {3, 218, 192}, {1, 220, 188}, {0, 222, 185}, {0, 225, 181},
    {0, 227, 178}, {0, 229, 174}, {0, 231, 170}, {0, 233, 166}, {1, 235, 161}, {3, 237, 157}, {6, 239, 152}, {9, 240, 147},
    {12, 242, 142}, {16, 243, 137}, {20, 245, 131}, {24, 246, 126}, {28, 247, 120}, {32, 248, 114}, {36, 249, 108}, {40, 249, 102},
    {44, 250, 96}, {48, 251, 90}, {52, 251, 84}, {56, 252, 78}, {60, 252, 72}, {63, 252, 66}, {67, 252, 60}, {70, 252, 55},
    {73, 252, 49}, {76, 252, 44}, {79, 251, 39}, {82, 251, 34}, {84, 251, 30}, {87, 250, 25}, {89, 250, 21}, {91, 249, 18},
    {93, 249, 15}, {95, 248, 12}, {97, 247, 10}, {99, 246, 8}, {100, 246, 7}, {102, 245, 6}, {103, 244, 6}, {105, 243, 6},
    {106, 242, 7}, {107, 241, 8}, {108, 240, 9}, {109, 239, 11}, {110, 238, 12}, {111, 237, 14}, {112, 236, 16}, {113, 235, 18},
    {113, 234, 20}, {114, 233, 22}, {115, 232, 24}, {115, 231, 26}, {116, 230, 29}, {116, 229, 31}, {117, 228, 33}, {117, 227, 35},
    {118, 226, 38}, {118, 225, 40}, {118, 224, 42}, {119, 222, 45}, {119, 221, 47}, {119, 220, 49}, {120, 219, 52}, {120, 218, 54},
    {120, 217, 56}, {120, 216, 59}, {121, 214, 61}, {121, 213, 63}, {121, 212, 66}, {121, 211, 68}, {121, 209, 70}, {122, 208, 73},
    {122, 207, 75}, {122, 206, 77}, {122, 204, 80}, {122, 203, 82}, {122, 202, 84}, {122, 200, 86}, {122, 199, 89}, {122, 198, 91},
    {122, 196, 93}, {122, 195, 96}, {121, 194, 98}, {121, 192, 100}, {121, 191, 102}, {121, 189, 105}, {120, 188, 107}, {120, 187, 109},
    {120, 185, 111}, {119, 184, 113}, {119, 182, 115}, {118, 181, 117}, {118, 180, 119}, {117, 178, 122}, {117, 177, 124}, {116, 175, 126},
    {115, 174, 128}, {115, 173, 130}, {114, 171, 132}, {113, 170, 134}, {113, 168, 136}, {112, 167, 138}, {111, 165, 140}, {110, 164, 142},
    {110, 163, 144}, {109, 161, 145}, {108, 160, 147}, {107, 158, 149}, {106, 157, 151}, {105, 156, 153}, {104, 154, 154}, {103, 153, 156},
    {102, 151, 158}, {101, 150, 160}, {100, 148, 161}, {99, 147, 163}, {98, 146, 165}, {97, 144, 166}, {96, 143, 168}, {95, 141, 170},
    {93, 140, 171}, {92, 139, 173}, {91, 137, 174}, {90, 136, 176}, {88, 135, 177}, {87, 133, 179}, {85, 132, 180}, {84, 131, 182},
    {82, 129, 183}, {81, 128, 185}, {79, 127, 186}, {78, 125, 187}, {76, 124, 189}, {74, 123, 190}, {73, 121, 191}, {71, 120, 193},
    {69, 119, 194}, {67, 117, 195}, {66, 116, 196}, {64, 115, 198}, {62, 113, 199}, {60, 112, 200}, {58, 111, 201}, {56, 109, 202},
    {54, 108, 203}, {52, 107, 204}, {50, 105, 205}, {48, 104, 206}, {46, 102, 207}, {44, 101, 207}, {42, 100, 208}, {40, 98, 209},
    {37, 97, 209}, {35, 96, 210}, {33, 94, 210}, {31, 93, 211}, {28, 91, 211}, {26, 90, 212}, {24, 89, 212}, {22, 87, 213},
    {20, 86, 213}, {18, 84, 213}, {16, 83, 214}, {14, 82, 214}, {12, 80, 214}, {10, 79, 214}, {8, 77, 214}, {6, 76, 214},
    {4, 74, 214}, {3, 73, 214}, {1, 71, 214}, {0, 70, 213}
  };
  std::array<cv::Vec3b, 256> depth_colormap_ = []() {
    std::array<cv::Vec3b, 256> cm;
    for (int i = 0; i < 256; ++i) {
      cm[i] = cv::Vec3b(depth_colormap_data_[i][2], depth_colormap_data_[i][1], depth_colormap_data_[i][0]);  // BGR order
    }
    return cm;
  }();

  // =========================================================================
  // Members
  // =========================================================================

  // --- parameters ---
  int    queue_size_{3};
  int    point_radius_{2};
  double max_distance_{8.0};

  // --- cameras (index 0 = front; index 1 = back, when configured) ---
  std::vector<Camera> cameras_;

  // --- the one thing both cameras share besides the parameters ---
  rclcpp::Subscription<CloudMsg>::SharedPtr  cloud_direct_sub_;

  // --- guards every camera's cloud_deque and colored_points_2d ---
  std::mutex  state_mutex_;
};

// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LidarCameraFusionNode>());
  rclcpp::shutdown();
  return 0;
}
