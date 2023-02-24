cmake_minimum_required(VERSION 2.8.3)
project(audio_to_spectrogram)

find_package(
  catkin REQUIRED
  catkin_virtualenv
  dynamic_reconfigure
)

if (${catkin_virtualenv_VERSION} VERSION_LESS "0.5.1")
  message(STATUS "jsk_rosbag_tools requires catkin_virtualenv >= 0.5.1")
  return()
endif()

catkin_python_setup()

generate_dynamic_reconfigure_options(
  cfg/AudioAmplitudePlot.cfg
)

catkin_package(
  CATKIN_DEPENDS
)

if($ENV{ROS_DISTRO} STREQUAL "noetic")
  catkin_generate_virtualenv(
    INPUT_REQUIREMENTS requirements.in
    PYTHON_INTERPRETER python3
  )
else()
  catkin_generate_virtualenv(
    INPUT_REQUIREMENTS requirements.in
  )
endif()

file(GLOB SCRIPTS_FILES scripts/*)
catkin_install_python(
  PROGRAMS ${SCRIPTS_FILES}
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch sample scripts test
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
)

install(FILES requirements.txt
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  find_package(catkin REQUIRED COMPONENTS rostest roslaunch)
  add_rostest(test/audio_to_spectrogram.test)
  roslaunch_add_file_check(launch/audio_to_spectrogram.launch)
endif()
