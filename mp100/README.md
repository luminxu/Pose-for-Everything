MP-100 is built upon the 2D pose datasets, including 
[COCO](http://cocodataset.org/), 
[300W](https://ibug.doc.ic.ac.uk/resources/300-W/), 
[AFLW](https://www.tugraz.at/institute/icg/research/team-bischof/learning-recognition-surveillance/downloads/aflw), 
[OneHand10K](https://www.yangangwang.com/papers/WANG-MCC-2018-10.html), 
[DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/LandmarkDetection.html), 
[AP-10K](https://github.com/AlexTheBad/AP-10K), 
[MacaquePose](http://www.pri.kyoto-u.ac.jp/datasets/macaquepose/index.html), 
[Vinegar Fly](https://github.com/jgraving/DeepPoseKit-Data), 
[Desert Locust](https://github.com/jgraving/DeepPoseKit-Data), 
[CUB-200](http://www.vision.caltech.edu/datasets/cub_200_2011/), 
[CarFusion](http://www.cs.cmu.edu/~ILIM/projects/IMgit/CarFusion/cvpr2018/index.html), 
[AnimalWeb](https://fdmaproject.wordpress.com/author/fdmaproject/), 
[Keypoint-5](https://github.com/jiajunwu/3dinn).
In order to use MP-100, please download images from the original datasets first, 
then reorganize the data and use our provided 
[annotation files](https://drive.google.com/drive/folders/1pzC5uEgi4AW9RO9_T1J-0xSKF12mdj1_?usp=sharing) for training and testing.
After preparing images and annotations, the project should look like this:

```text
Pose-for-Everything
├── assets
├── configs
├── mp100
├── pomnet
├── tools
`── data
    │── mp100
        │-- annotations
        │   │-- mp100_split1_train.json
        │   |-- mp100_split1_val.json
        │   |-- mp100_split1_test-dev-2017.json
        │   │-- ...
        │-- human_face
        │-- human_hand
        │-- sling_dress
        │-- human_body
        │   │-- 000000000009.jpg
        │   │-- 000000000025.jpg
        │   │-- 000000000030.jpg
        │   │-- ...
        │-- antelope_body
        │-- ...

```


MP-100 includes 100 categories and the images of different categories are contained in different folders individually. 
Specifically, 

- **human_body** is collected from [COCO](http://cocodataset.org/).

    Here is an example that soft link can be established from the downloaded images to propare the data for each category.

    ```shell
    ln -s ${COCO_PATH} data/mp100/human_body
    ```

- **human_face** is collected from [300W](https://ibug.doc.ic.ac.uk/resources/300-W/).

- **amur_tiger_body** is collected from [AFLW](https://www.tugraz.at/institute/icg/research/team-bischof/learning-recognition-surveillance/downloads/aflw).

- **human_hand** is collected from [OneHand10K](https://www.yangangwang.com/papers/WANG-MCC-2018-10.html).

- 13 categories including **long_sleeved_dress**, 
**long_sleeved_outwear**, 
**long_sleeved_shirt**, 
**shorts**, 
**short_sleeved_dress**, 
**short_sleeved_outwear**, 
**short_sleeved_shirt**, 
**skirt**, 
**sling**, 
**sling_dress**, 
**trousers**, 
**vest**, 
and **vest_dress** 
are collected from [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/LandmarkDetection.html).

    For simplicity, all the categories can be linked to the complete downloaded dataset.
    
    ```shell
    ln -s ${DEEPFASHION_DATA} data/mp100/long_sleeved_dress
    ln -s ${DEEPFASHION_DATA} data/mp100/long_sleeved_outwear
    ln -s ${DEEPFASHION_DATA} data/mp100/long_sleeved_shirt
    ...
    ```

- 34 categories including
**antelope_body**, 
**beaver_body**, 
**bison_body**, 
**bobcat_body**, 
**cat_body**, 
**cheetah_body**, 
**cow_body**, 
**deer_body**, 
**dog_body**, 
**elephant_body**, 
**fox_body**, 
**giraffe_body**, 
**gorilla_body**, 
**hamster_body**, 
**hippo_body**, 
**horse_body**, 
**leopard_body**, 
**lion_body**, 
**otter_body**, 
**panda_body**, 
**panther_body**, 
**pig_body**, 
**polar_bear_body**, 
**rabbit_body**, 
**raccoon_body**, 
**rat_body**, 
**rhino_body**, 
**sheep_body**, 
**skunk_body**, 
**spider_monkey_body**, 
**squirrel_body**, 
**weasel_body**, 
**wolf_body**, 
and **zebra_body**
are collected from [AP-10K](https://github.com/AlexTheBad/AP-10K).

- **macaque_body** is collected from [MacaquePose](http://www.pri.kyoto-u.ac.jp/datasets/macaquepose/index.html).

- **fly_body** is collected from [Vinegar Fly](https://github.com/jgraving/DeepPoseKit-Data).

- **locust_body** is collected from [Desert Locust](https://github.com/jgraving/DeepPoseKit-Data).

- 8 categories are collected from [CUB-200](http://www.vision.caltech.edu/datasets/cub_200_2011/). 
In detail, 

    **grebe_body** is the combination of 050.Eared_Grebe, 
    051.Horned_Grebe, 052.Pied_billed_Grebe, and 053.Western_Grebe in CUB-200.
    
    **gull_body** is the combination of 059.California_Gull, 
    060.Glaucous_winged_Gull, 061.Heermann_Gull, 062.Herring_Gull, 
    063.Ivory_Gull, 064.Ring_billed_Gull, 065.Slaty_backed_Gull, 
    and 066.Western_Gull in CUB-200.
    
    **kingfisher_body** is the combination of 079.Belted_Kingfisher, 
    080.Green_Kingfisher, 081.Pied_Kingfisher, 082.Ringed_Kingfisher, 
    and 083.White_breasted_Kingfisher in CUB-200.
    
    **sparrow_body** is the combination of 113.Baird_Sparrow, 114.Black_throated_Sparrow, 
    115.Brewer_Sparrow, 116.Chipping_Sparrow, 117.Clay_colored_Sparrow, 
    118.House_Sparrow, 119.Field_Sparrow, 120.Fox_Sparrow, 
    121.Grasshopper_Sparrow, 122.Harris_Sparrow, 123.Henslow_Sparrow, 
    124.Le_Conte_Sparrow, 125.Lincoln_Sparrow, 126.Nelson_Sharp_tailed_Sparrow, 
    127.Savannah_Sparrow, 128.Seaside_Sparrow, 129.Song_Sparrow, 
    130.Tree_Sparrow, 131.Vesper_Sparrow, 132.White_crowned_Sparrow, 
    and 133.White_throated_Sparrow in CUB-200.

    **tern_body** is the combination of 141.Artic_Tern, 142.Black_Tern, 
    143.Caspian_Tern, 144.Common_Tern, 145.Elegant_Tern, 
    146.Forsters_Tern, and 147.Least_Tern in CUB-200.
    
    **warbler_body** is the combination of 158.Bay_breasted_Warbler, 159.Black_and_white_Warbler, 
    160.Black_throated_Blue_Warbler, 161.Blue_winged_Warbler, 162.Canada_Warbler, 
    163.Cape_May_Warbler, 164.Cerulean_Warbler, 165.Chestnut_sided_Warbler, 
    166.Golden_winged_Warbler, 167.Hooded_Warbler, 168.Kentucky_Warbler, 
    169.Magnolia_Warbler, 170.Mourning_Warbler, 171.Myrtle_Warbler, 
    172.Nashville_Warbler, 173.Orange_crowned_Warbler, 174.Palm_Warbler, 
    175.Pine_Warbler, 176.Prairie_Warbler, 177.Prothonotary_Warbler, 
    178.Swainson_Warbler, 179.Tennessee_Warbler, 180.Wilson_Warbler, 
    181.Worm_eating_Warbler, and 182.Yellow_Warbler in CUB-200.
    
    **woodpecker_body** is the combination of 187.American_Three_toed_Woodpecker, 
    188.Pileated_Woodpecker, 189.Red_bellied_Woodpecker, 190.Red_cockaded_Woodpecker, 
    191.Red_headed_Woodpecker, and 192.Downy_Woodpecker in CUB-200.
    
    **wren_body** is the combination of 193.Bewick_Wren, 
    194.Cactus_Wren, 195.Carolina_Wren, 196.House_Wren, 
    197.Marsh_Wren, 198.Rock_Wren, and 199.Winter_Wren in CUB-200.
    
    As the images of the category come from multiple sources, we can copy or move all the needed images to the new folder.
    For example, 
    
    ```shell
    mkdir grebe_body
    
    # copy images to the new folder
    cp ${CUB-200_ROOT}/*_Grebe/* data/mp100/grebe_body
    or
    # move images to the new folder
    mv ${CUB-200_ROOT}/*_Grebe/* data/mp100/grebe_body
    ```

- 3 categories including 
**bus**, 
**car**, 
and **suv**
are collected from [CarFusion](http://www.cs.cmu.edu/~ILIM/projects/IM/CarFusion/cvpr2018/index.html).
We clean the data and select the samples manually. 
Also, we rename the images to image_id.jpg using [rename_carfusion_image.py](rename_carfusion_image.py). 
*image_id* is the ID of each image in COCO format obtained by the [official tools](https://github.com/dineshreddy91/carfusion_to_coco).

    First, we can use the code provided by [official tools](https://github.com/dineshreddy91/carfusion_to_coco) to 
    convert the annotations to COCO format. 
    Then, we run [rename_carfusion_image.py](rename_carfusion_image.py) to rename the images.
    
    ```shell
    python mp100/rename_carfusion_image.py --ann_file ${COCO_FORMAT_ANNOTATION} \
      --img_src ${CARFUSION_DATA} --write_dir data/mp100
    ```

- 30 categories including 
**alpaca_face**, 
**arcticwolf_face**, 
**bighornsheep_face**, 
**blackbuck_face**, 
**bonobo_face**, 
**californiansealion_face**, 
**camel_face**, 
**capebuffalo_face**, 
**capybara_face**, 
**chipmunk_face**, 
**commonwarthog_face**, 
**dassie_face**, 
**fallowdeer_face**, 
**fennecfox_face**, 
**ferret_face**, 
**gentoopenguin_face**, 
**gerbil_face**, 
**germanshepherddog_face**, 
**gibbons_face**, 
**goldenretriever_face**, 
**greyseal_face**, 
**grizzlybear_face**, 
**guanaco_face**, 
**klipspringer_face**, 
**olivebaboon_face**, 
**onager_face**, 
**pademelon_face**, 
**proboscismonkey_face**, 
**przewalskihorse_face**, 
and **quokka_face** 
are collected from [AnimalWeb](https://fdmaproject.wordpress.com/author/fdmaproject/).

- 5 categories including 
**bed**, 
**chair**, 
**sofa**, 
**swivelchair**, 
and **table** 
are collected from [Keypoint-5](https://github.com/jiajunwu/3dinn).





## Citation

If you find this dataset useful in your research, please consider cite:

```bibtex
@article{xu2022pose,
  title={Pose for Everything: Towards Category-Agnostic Pose Estimation},
  author={Xu, Lumin and Jin, Sheng and Zeng, Wang and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping and Wang, Xiaogang},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022},
  month={October}
}
```
