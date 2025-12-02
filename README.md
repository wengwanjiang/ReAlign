<h1 align="center"><strong>ReAlign: Text-to-Motion Generation via Step-Aware Reward-Guided Alignment</strong></h1>
<p align="center">
 <a href='https://github.com/wengwanjiang' target='_blank'>Wanjiang Weng<sup>*</sup></a>&emsp;
 <a href='https://xiaofeng-tan.github.io/' target='_blank'>Xiaofeng Tan<sup>*</sup></a>&emsp;
 Junbo Wang&emsp;
 Guo-Sen Xie&emsp;
 Pan Zhou&emsp;
 Hongsong Wang<sup>†</sup>&emsp;
  <br>
  *Equal Contribution&emsp;
  †Corresponding Author
</p>

<p align="center">
  <a href="https://aaai.org/conference/aaai/aaai-26/">
    <img src="https://img.shields.io/badge/AAAI-2026-138D75" alt="AAAI 2026">
  </a>
  <a href="https://arxiv.org/abs/2511.19217">
    <img src="https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow" alt="Paper PDF on arXiv">
  </a>
 <a href='https://wengwanjiang.github.io/ReAlign-page'>
  <img src='https://img.shields.io/badge/Project-Page-%23df5b46?style=flat&logo=Google%20chrome&logoColor=%23df5b46'></a> 
</p>

> **TL;DR:** We propose **ReAlign**, a *plug-and-play reward-guided alignment strategy* for text-to-motion generation, which explicitly enhances both semantic consistency and motion realism throughout the denoising process.

This repository offers the official Pytorch code for this paper. The code will be released before the conference of AAAI-26.


If you have any questions, feel free to contact Wanjiang Weng (wjweng@seu.edu.cn) or Xiaofeng Tan (xiaofengtan@seu.edu.cn).

## 📣 News
- **[2025/11]** The paper has been publicly released.
- **[2025/11]** **ReAlign** has been officially accepted by *AAAI 2026*! 🎉

## 📆 Plan
- [x] Release early version.
- [x] Release [final version](https://arxiv.org/abs/2511.19217).
- [ ] Release code for T2M: 
  - [ ] Release environment guidance.
  - [x] Release evaluation code.
  - [ ] Release inference code.
  - [x] Release training code.
  - [x] Release pretrained model weights.



## Model Zoo
<table>
  <tr>
    <th>Model Name</th>
    <th>Dataset</th>
    <th>Download Link</th>
    <th>Retrieval Performance (R@1)</th>
  </tr>
  <tr>
    <td rowspan="2">Step-Aware Reward Model</td>
    <td>HumanML3D</td>
    <td>
      <a href="https://1drv.ms/">OneDrive</a>,
      <a href="https://pan.baidu.com/s/1HHux8t_cCaENw9_ybrGIOg">BaiduNetDisk (passwd: 1234)</a>
    </td>
    <td>T2M: 67.59%, M2T: 68.94%</td>
  </tr>
  <tr>
    <td>KIT-ML</td>
    <td>
      <a href="https://1drv.ms/">OneDrive</a>,
      <a href="https://pan.baidu.com/s/1HHux8t_cCaENw9_ybrGIOg">BaiduNetDisk (passwd: 1234)</a>
    </td>
    <td>T2M: 52.84%, M2T: 52.98%</td>
  </tr>
</table>

## Citation

  ```

@inproceedings{wengReAlign26,
  title={ReAlign: Text-to-Motion Generation via Step-Aware Reward-Guided Alignment}, 
  author={Wanjiang Weng and Xiaofeng Tan and Junbo Wang and Guo-Sen Xie and Pan Zhou and Hongsong Wang},
  year={2025},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence}
}
  ```
