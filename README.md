# PyTorch implementation of DistillGCN

Paper: [Distilling Knowledge From Graph Convolutional Networks](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Distilling_Knowledge_From_Graph_Convolutional_Networks_CVPR_2020_paper.html), CVPR'20

## Method Overview

![](./asserts/overview.png)

## Dependencies

### Main packages

> PyTorch = 1.1.0

> DGL = 1.4.0

See [requirment](requirments.txt) file for more information
about how to install the dependencies.

## Training and evaluation

The [main.py](main.py) file contains the code for training teacher model, training the student model using the LSP module.

Early stop is used when training both the student model and the teacher model.

## Cite

```
@inproceedings{yang2020distilling,
  title={Distilling Knowledge From Graph Convolutional Networks},
  author={Yang, Yiding and Qiu, Jiayan and Song, Mingli and Tao, Dacheng and Wang, Xinchao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7074--7083},
  year={2020}
}
```

## License

DistillGCN is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.