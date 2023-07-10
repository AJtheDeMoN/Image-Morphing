# Image Morphing Project

This is a README file for the Image Morphing project. This project aims to perform image morphing, a technique that creates a smooth transition between two images by blending corresponding pixels. By generating a sequence of intermediate images, this technique can create visually appealing transformations between two different objects or faces.

## Features

- Morphing between two input images
- Control over the morphing process through adjustable parameters
- Generating a sequence of intermediate frames
- Saving the morphed frames as a video or individual images

## Requirements

To run the Image Morphing project, you need the following dependencies:

- Python (version 3.6 or later)
- OpenCV (version 4.0 or later)
- NumPy (version 1.19 or later)
- Matplotlib (version 3.2 or later)

## Installation

1. Clone the project repository:

   ```
   git clone https://github.com/AJtheDeMoN/Image-Morphing/
   ```

## Usage

1. Place your input images in the `input` directory of the project.

2. Run the `Image_Morphing.py` script:

   ```
   python Image_Morphing.py 
   ```

   Replace `image1.jpg` and `image2.jpg` with the filenames of your input images. The `--output` flag specifies the path and filename of the generated morphed frames.

   Additional options are available to control the morphing process, such as `--num_frames` to specify the number of intermediate frames and `--output_format` to choose the output format (`gif` or `video`).

3. Once the script finishes executing, you can find the morphed frames in the `output` directory.

## Acknowledgments

This Image Morphing project was inspired by the concept of image morphing and its applications in computer graphics. The implementation builds upon the techniques and ideas from various research papers and online resources.
