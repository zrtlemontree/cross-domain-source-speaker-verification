# A self-distillation-based domain exploration framework for cross-voice-transfer attacker verification

This is the official implementation of the data processing in the paper [**A self-distillation-based domain exploration framework for cross-voice-transfer attacker verification**](). In this paper, we poposed the SDDE for cross-voice-transfer attacker verification. Besides, we constructed a large-scale attacker verification data set (over 7,945 hours) using three well-known voice conversion algorithms and speaker verification corpora.


You can find the conversion lists in manifest/, and the test file is available [here](https://drive.google.com/drive/folders/1n2pREZfZh9CRo7GSBaQZ0X8cbHg-l2Jk?usp=sharing).

The figure shows the data processing overview. We adopted the LibriSpeech as the attacker data set and the VoxCeleb as the victim data set. For each speech from the victim data set was attacked by three attackers' speech, generating three spoofed speech.

<img src="./fig/data processing.png" width="400">


## Getting started
### Requirements
Python 3.8.17 is used, other requirements are listed in requirements.txt:
```bash
pip install -r requirements.txt
```

### AGAIN-VC
1. Prepare the [pre-trained model](https://github.com/KimythAnly/AGAIN-VC) and put it in againvc/pretrain.

2. Modify the config/againvc.yaml according to your needs.

3. Execute the following command.

```bash
python main.py --config ./config/againvc.yaml
```

### FreeVC

1. Download [WavLM-Large](https://github.com/microsoft/unilm/tree/master/wavlm) and put it under directory freevc/wavlm/

2. Prepare the [pre-trained model](https://github.com/OlaWod/FreeVC) of FreeVC and put it in freevc/checkpoints.

<!-- 3. Execute the following command. -->

```bash
python main.py --config ./config/freevc.yaml
```

### VQMIVC

1. Install [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN) 

2. Download the [pre-trained models](https://github.com/Wendison/VQMIVC) of VQMIVC and put them in vqmivc/checkpoints and vqmivc/vocoder

<!-- 3. Execute the following command. -->

```bash
python main.py --config ./config/vqmivc.yaml
```

## References

- https://github.com/KimythAnly/AGAIN-VC
- https://github.com/OlaWod/FreeVC
- https://github.com/Wendison/VQMIVC
- https://github.com/kan-bayashi/ParallelWaveGAN
- https://github.com/microsoft/unilm/tree/master/wavlm
- https://github.com/descriptinc/melgan-neurips

## License
This project is mainly licensed under the MIT License ( ./LICENSE ). Each folder within the project maycontain their corresponding LlCENSE according to the external libraries used. Please refer to the README.md filein each folder for more details.

Additionally, specific licenses for some of the external libraries used are mentioned below:

- ./againvc is licensed MIT License (./againvc/LICENSE)
- ./againvc/descriptinc/melgan-neurips-master is licensed MIT License (./againvc/descriptinc/melgan-neurips-master/LICENSE)
- ./freevc is licensed MIT License (./freevc/LICENSE)
- ./vqmivc is licensed MIT License (./vqmivc/LICENSE)
- ./vqmivc/ParallelWaveGAN is licensed MIT License (./vqmivc/ParallelWaveGAN/LICENSE)