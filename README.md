# HFPAC (High-Fidelity Python Audio Codec)

## What is HFPAC?
HFPAC stands for **High-Fidelity Python Audio Codec**. It is an audio compression format designed from the ground up to prioritize maximum audio quality (high-fidelity) while still applying as much compression as possible to reduce file sizes.

## Repository Contents
This repository contains the two core tools needed to interact with the HFPAC format:
* **The Encoder (`encoder_gui.py`):** A tool (complete with a Graphical User Interface) to efficiently compress your standard audio files into the `.hfpac` format.
* **The Player (`gui.py` / `player.py`):** A dedicated playback application and GUI for opening, decoding, and listening to `.hfpac` encoded audio files.

## Versioning System
The project's versioning is split into three distinct categories to easily track updates:
* **Player & GUI Updates:** Uses the format `6.0.x.y` (e.g., 6.1.4.3). Tracks improvements, bug fixes, and feature additions to the playback software.
* **Encoder & GUI Updates:** Uses the format `6.0.x.y` (e.g., 6.1.2.0). Tracks performance upgrades, UI enhancements, and optimizations for the compression software.
* **HFPAC Format Updates:** Uses the format `vX.X` (e.g., v6.1, 6.2, 6.3). Tracks fundamental changes to the underlying codec algorithms and file structure.

## Best Use Cases
* **High-Fidelity Archival:** Perfect for archiving audio where loss of quality is unacceptable, but storage space still needs to be heavily optimized.
* **Audiophile Listening:** For end-users who want to maintain the absolute highest fidelity of their music library.
* **Python Audio Processing:** Ideal for developers looking to integrate a high-quality native Python audio codec into their own applications or pipelines.

## How HFPAC Works
HFPAC achieves its high-fidelity compression by leveraging a multi-stage encoding pipeline mathematically optimized for audio signals. Based on the source files, the core pipeline utilizes:
1. **LPC (Linear Predictive Coding):** Analyzes the audio signal to estimate future samples based on previous ones, separating the predictable signal from the unpredictable "residual" noise.
2. **Rice Coding:** Applies efficient entropy encoding to the residual signals, which are typically small and well-suited for Rice compression.
3. **Huffman Coding:** An additional layer of lossless entropy encoding that analyzes the frequency of the generated data to further compress the final output into the `.hfpac` file format.
