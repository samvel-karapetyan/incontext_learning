# Codebase documentation

## Setup Instructions
These instructions will guide you through the process of setting up the project environment. Please follow each step carefully.

1. **Clone the Repository**: Start by cloning the repository to your local machine. 
   ```commandline
   git clone git@github.com:YerevaNN/incontext_spurious.git
   cd incontext_spurious
   ```
2. **Configure Environment Variables:**
   - Copy the sample environment file:	
     ```commandline
     cp .env.sample .env
     ```
   - Open the `.env` file and fill in the necessary environment variables as specified in the file.
3. **Create a Conda Environment:**
   - Create a new Conda environment using the provided environment file:
     ```commandline
     conda env create
     ```
   - Activate the new environment:
     ```commandline
     conda activate incontext_spurious
     ```
4. **Initialize Aim logger:**
   ```commandline
   aim init
   ```

## Data Download and Preparation Instructions
**Overview:**

The dataset required for this process will be automatically downloaded during the encoding extraction run, eliminating the need for manual downloading!

1. **Extracting and Saving Encodings**

    To begin, run the script for extracting and saving encodings. By default, this uses the `dinov2_vitb14` configuration.
    
    ```commandline
    python run.py --config-name=extract_encodings
    ```

2. **Computing Average Norm and Generating Tokens**
    
    This step involves computing the average norm of encoding vectors. It also generates fixed tokens. During training, you have the option to use these fixed tokens or generate new ones for each instance.
    
    ```commandline
    python run.py --config-name=compute_encodings_avg_norm_and_generate_tokens
    ```

### ❗❗❗❗❗❗❗ Attention ❗❗❗❗❗❗❗

- Point 2 should only be executed by one individual to ensure consistency in the validation sets.
- These steps have already been completed. The generated files (avg_norms and context_val_sets) are available. Team members can access their location via the Notion documentation.


## Training

This section provides instructions for running the training script, using [Hydra](https://hydra.cc/), a Python library, for configuration management. Hydra allows for flexible and powerful configuration, enabling you to modify settings directly from the command line.

### **Running the Training Script**

```commandline
python run.py
```

### **Customizing Configurations with Hydra**

Hydra configurations provide a flexible way to adjust training parameters. You can modify these configurations either directly in the configuration files or via the command line.

- **Reviewing Configurations in Files:**
    - Configuration files are located in the `configs` folder.
    - The `train.yaml` file is the root configuration file for the training script.
- **Command Line Configuration Overrides (Recommended):**
    - Hydra allows you to override configurations directly from the command line, which is the recommended approach (alternatively, you can modify the config files).
    - This method is quick and does not require modifying the configuration files directly.
    
    **Example Command:**
    
    ```bash
    python run.py optimizer=adam optimizer.lr=0.01
    ```
    
    In this example, the `optimizer` is set to `adam`, and the learning rate (`lr`) is set to `0.01`.