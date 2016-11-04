BoundingBoxFilter
================

.. image:: images/bounding_box_filter.png


What is this?
-------------



Subscribing Topics
------------------

- ``~input_box`` (``jsk_recognition_msgs/BoundingBoxArray``)

- ``~input_indices`` (``jsk_recognition_msgs/ClusterPointIndices``)

Publishing Topics
-----------------

- ``~output_box`` (``jsk_recognition_msgs/BoundingBoxArray``)

- ``~output_indices`` (``jsk_recognition_msgs/ClusterPointIndices``)

Parameters
----------

- ``~vital_rate`` (``Double``, default: ``1.0``)

- ``~use_x_dimension`` (``Bool``, default: ``False``)

  Use x dimension to filter

- ``~x_dimension_min`` (``Double``, default: ``0.1``)

  Minimum value for x dimension

- ``~x_dimension_max`` (``Double``, default: ``0.1``)

  Maximum value for x dimension

- ``~use_y_dimension`` (``Bool``, default: ``False``)

  Use y dimension to filter

- ``~y_dimension_min`` (``Double``, default: ``0.1``)

  Minimum value for y dimension

- ``~y_dimension_max`` (``Double``, default: ``0.1``)

  Maximum value for y dimension

- ``~use_z_dimension`` (``Bool``, default: ``False``)

  Use z dimension to filter

- ``~z_dimension_min`` (``Double``, default: ``0.1``)

  Minimum value for z dimension

- ``~z_dimension_max`` (``Double``, default: ``0.1``)

  Maximum value for z dimension

- ``~filter_limit_negative`` (``Bool``, default: ``False``)

  Set to true if we want to return the data outside [filter_limit_min; filter_limit_max]
