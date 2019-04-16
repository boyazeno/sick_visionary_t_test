^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package sick_visionary_t_driver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.0.2 (2016-10-07)
------------------
* rename to visionary_t
* Contributors: Florian Weisshardt

0.0.1 (2016-08-02)
------------------
* fixed warning
* added parameter:prevent_frame_skipping
  f true: prevents skipping of frames and publish everything, otherwise use newest data to publish to ROS world
* using pointer instead of copy constructor (optimization)
* warning instead of debug msg
* fixed receive queue
* added functions to check header + size before parsing
* debug messaging
* check data header
* detach publishing data from network thread
* Renamed launch file to match the driver name
* Updated formatting and adding parameter comment in launch file.
* initial version of sick_visionary_t_driver ROS driver
* Contributors: Florian Weisshardt, Joshua Hampp, Marco Dierschke
