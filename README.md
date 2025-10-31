# Corrosion Quantification Project

This repository contains scripts and data related to the quantification of corrosion using image processing and machine learning techniques. Below is a brief description of each file and its purpose:

## Files

### Scripts for classification pipeline
- **IBEXtrainingLocal.py**: Script for training on a GPU Cluster. 
- **distributedTrainingLocal.py**: Script for distributed training of the corrosion detection model locally. This is the one to use on a normal laptop.
- **iqToRichImages.py**: Script for converting IQ samples to images (Fast Fourier Transform stacks)
- **resnet_corrosion.pth**: Pre-trained ResNet model weights specifically tuned for corrosion detection tasks.
- **splitBalanced.py**: Script for splitting the dataset into balanced training and testing sets to ensure the model performs well on various types of corrosion.
- **splitUnbalanced.py**: Script for splitting the dataset into unbalanced training and testing sets.

### Additional scripts for performance improvements

- **thermalImpact.py**: Data augmentation to test robustness of the model to temperature changes, including in out-of-distribution testing
-  ** SNR.py **: Compute signal-to-noise ration on collected .iq files at 5 cm and 25 cm distance.
-  **SNR_pipe.py**: Same as above but for the pipe files.
-  **frequency_tuning.py**: Compute dBFS over .iq files collected with different frequencies, to highlight the most receptive frequencies for each oxide type.


### Data
- **matrices**: Directory containing matrix data used for training and testing models.
- **results**: Directory containing the results of the corrosion quantification analysis.
- **Image Database**: Directory containing all grayscale images derived from IQ samples. Images were generated with an FFT size of 4096 and 128 stacks. Storing images in a very lightweight format, but does not provide the versatility of storing IQ samples.

### Simulation Meep
- **foil_0.5.py**: Simulation for one oxide in a pipe
- **foil_spread_oxidesLB.py**: Simulation for five oxides in a pipe
- **sim_without_pipeLB.py**: Simulation for one oxide in open air
-  **sim_without_pipe_2_5gLB.py**: Simulation for five oxides in open air

## Usage

0. **Import IQ samples**: Import the IQ samples, ideally in the samples directory as the scripts are pointing there.
1. **iqToRichImages**: Take the IQ samples, apply FFT, and stack them to create grayscale images.
2. **Training**: Use `IBEXtrainingLocal.py` or `distributedTrainingLocal.py` to train the model on your data. The script will also test the resulting model and provide usual ML metrics. Typical behavior will be similarity between the metrics, the most relevant being F1-score in corrosion detection. 

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
