cmake_minimum_required(VERSION 2.8.3)
project(sound_localization)

find_package(catkin REQUIRED
  catkin_virtualenv
  std_msgs
  roslaunch
  )
if (${catkin_virtualenv_VERSION} VERSION_LESS "0.3.0")
  message(STATUS "sound_localization requires catkin_virtualenv >= 0.3.0")
  return()
endif()

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
)

catkin_generate_virtualenv(
  INPUT_REQUIREMENTS requirements.in
  PYTHON_INTERPRETER python3
)

file(GLOB NODE_SCRIPTS_FILES node_scripts/*.py)
catkin_install_python(
  PROGRAMS ${NODE_SCRIPTS_FILES}
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch scripts sample_rosbag
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
  )

install(FILES requirements.txt
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  find_package(catkin REQUIRED COMPONENTS roslaunch rostest)
endif()
