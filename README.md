
# HSI_depth

A Python tool to perform  experiments on various hyperspectral datasets in 3d vision and\or depth sphere.
https://www.ximea.com/support/wiki/apis/Python_inst_win
Sample result output https://github.com/air55555/HSI_depth/blob/master/3d_scan_sample_output.png

### 3d Mesh creation from 3 col file ###
https://gazebosim.org/api/gazebo/4.0/pointcloud.html

Good results with flat meshes only, not cylinder.Example file - https://github.com/air55555/HSI_depth/blob/master/img/45_small/45smallMesh%5Bmkm_fast_middle_mass_1%2C4-5-1_3col%20-%20Cloud%5D%20(level%2010).obj
Could be opened with Windows built in 3d viewer 

Processing this input point cloud file https://github.com/air55555/HSI_depth/blob/master/img/45_small/mkm_fast_middle_mass_1%2C4-5-1_3col.csv
CloudCompare settings   https://github.com/air55555/HSI_depth/blob/master/img/45_small/normals.png  and https://github.com/air55555/HSI_depth/blob/master/img/45_small/poisson_reconstr.png

Bullet 3col file https://github.com/air55555/HSI_depth/blob/master/img/bullet_superQ_X0_704-Y0_960out/mkm_fast_middle_mass_1%2C4-5-1_3col.csv
Ruler 3col file https://github.com/air55555/HSI_depth/blob/master/img/ruler_X0_704-Y0_960out/mkm_fast_middle_mass_1%2C4-5-1_3col.csv

Python 3.6 due pptk limitation. 
Open3d issues possible https://github.com/isl-org/Open3D/issues/2136