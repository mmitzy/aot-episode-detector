# AOT Episode Detector

A deep learning project that classifies **Attack on Titan** episodes using a ResNet50 2-head architecture, with anomaly detection for identifying opening/ending frames and recaps.

## Overview

This project combines **image classification** and **anomaly detection** to:
- **Classify** which episode and season a given frame belongs to
- **Detect** opening/ending sequences and recap segments using anomaly detection techniques

The classifier uses a ResNet50 backbone with two output heads to simultaneously predict episode and season information from video frames.

## Features

- **ResNet50 2-Head Classifier**: Dual-output architecture for episode and season classification
- **Anomaly Detection**: Identifies opening/ending frames and recap sections using statistical anomaly detection
- **Season Support**: Covers Attack on Titan Seasons 1–4 plus Junior High and OAD (Original Animation Disc) content
- **Organized Dataset Structure**: Raw episode frames organized by season and content type

## Project Structure

```
aot-episode-detector/
├── AOT - RAW/                    # Raw episode data
│   ├── Junior High/              # Junior High season
│   │   ├── Episodes/
│   │   ├── Previews/
│   │   └── Trailers/
│   ├── OAD/                      # Original Animation Disc
│   │   └── Episodes/
│   ├── S1/ through S4/           # Main series seasons
│   │   ├── Episodes/
│   │   ├── Shorts/ (where applicable)
│   │   └── Trailers/
├── .venv/                        # Python virtual environment
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mmitzy/aot-episode-detector.git
   cd aot-episode-detector
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # On Windows PowerShell
   # or
   source .venv/bin/activate     # On macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

(Add usage examples and API documentation as the project develops)

## Model Architecture

### Classification Head
- **Input**: RGB video frames (preprocessed)
- **Backbone**: ResNet50 (pre-trained ImageNet weights)
- **Outputs**: 
  - Episode head: Classifies which episode (1–N per season)
  - Season head: Classifies season (Junior High, OAD, S1, S2, S3, S4)

### Anomaly Detection
Detects opening/ending frames and recap segments using:
- Frame-level feature extraction
- Statistical anomaly detection on extracted features
- Identifies unusual visual patterns characteristic of openings, endings, and recaps

## Dataset

The dataset can either be found here - *add link later
Or be extracted using extract_aot_frames.ps1 on the raw video files.

## Contributing

Contributions are welcome! Feel free to:
- Improve the model architecture
- Enhance anomaly detection
- Add support for additional content
- Optimize preprocessing pipelines

## License

This project is licensed under the MIT License. See `LICENSE` file for details.

## Acknowledgments

- Attack on Titan © Hajime Isayama
- ResNet architecture from [He et al., 2015](https://arxiv.org/abs/1512.03385)
