Stable Diffusion Image Generation
This repository contains scripts and tools to generate images using the Stable Diffusion model. It includes a script for running inference with a text prompt and a utility for converting original checkpoint (.ckpt) files into the Diffusers library format.


⚙️ Setup and Installation
Follow these steps to get the project running on your local machine.

1. Prerequisites
Make sure you have Python 3.8+ and Git installed.

2. Clone the Repository
Bash

git clone https://github.com/Faham-from-nowhere/Stable-diffusion.git
cd Stable-diffusion
3. Install Dependencies
It's recommended to use a virtual environment.

Bash

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required libraries
pip install torch diffusers transformers
4. Download the Model Weights
The main model file (.ckpt) is too large to be stored in this repository. You need to download it manually.

Download the v1-5-pruned-emaonly.ckpt model from the official Hugging Face repository.

Place the downloaded v1-5-pruned-emaonly.ckpt file inside the data/ directory of this project.

After this step, your data folder should look like this:
`
data/
├── merges.txt
├── model_converter.py
├── run_generation.py
├── special_tokens_map.json
├── tokenizer_config.json
├── v1-5-pruned-emaonly.ckpt  <-- The file you just downloaded
└── vocab.json
`


🚀 Usage
Image Generation
To generate an image from a text prompt, use the run_generation.py script.

Bash

python data/run_generation.py --prompt "a beautiful landscape painting by monet"
The generated image will be saved in the Images/ folder.

Model Conversion (Optional)
This repository also includes a script to convert .ckpt files to the Hugging Face Diffusers format.

Bash

python data/model_converter.py --checkpoint_path data/v1-5-pruned-emaonly.ckpt --dump_path ./converted_model
This will create a new folder named converted_model containing the converted model components.

🙏 Acknowledgements
This project is built upon the incredible work and open-source models from:

Stability AI
Umar Jamil
The Hugging Face team for the diffusers library.

📄 License
This project is licensed under the MIT License. See the LICENSE file for more details.
