
slam, simultaneous localization and mapping

- Environment config
  - Ubuntu 18.04 / Macbook
  - [OpenCV 4.5.2](https://github.com/opencv/opencv/tree/4.5.2)
  - [OpenCV Contrib 4.5.2](https://github.com/opencv/opencv_contrib/tree/4.5.2)
  - [Ceres 2.0.0](http://www.ceres-solver.org/installation.html)

SLAM

- 3D from

The normalized 8-point algorithm
```math
x_2Fx_1=0
```
```puml
@startuml
:GetData;
:EstimateTransformMatrix;
:BuildMap
@enduml
```

# matching
## sparse matching
## dense matching
