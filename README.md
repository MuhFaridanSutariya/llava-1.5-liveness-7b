# Llava-1.5-Liveness-7b : A Vision Language Model for Liveness Detection

- Team Leader: <b>[Firqa Aqila Noor Arasyi](https://github.com/firqaaa)</b>
- Team Member: <b>[Muhammad Hasnan Ramadhan](https://github.com/hasnanmr), [Muhammad Faridan Sutariya](https://github.com/MuhFaridanSutariya) and [Rasyidan Akbar Fayrussani](https://github.com/0xrsydn)</b>

## Vision Language Model Workflow for User Registration and Inference

![Statistics (1)](https://github.com/MuhFaridanSutariya/llava-1.5-liveness-7b/assets/88027268/15e0eccb-46a1-451a-8655-41a78769da50)


## Hardware Requirements for Efficient Model Inference

### RAM Requirements:
- **Minimum 32GB of RAM:** Provides sufficient memory to handle the large datasets and model parameters during inference, ensuring smooth and efficient data processing.

### VRAM Requirements:
- **Minimum 24GB of VRAM:** Ensures the GPU can effectively manage the high computational and memory demands of the model, facilitating faster and more efficient inference operations.

## How to run

### 1. Clone this repository
To get started, clone this repository onto your local machine. Follow the instructions below:

1. Open a terminal or Command Prompt.
2. Change to the directory where you want to clone the repository.
3. Enter the following command to clone the repository:
   ```bash
   git clone https://github.com/MuhFaridanSutariya/llava-1.5-liveness-7b.git
   ```
4. Once the cloning process is complete, navigate into the cloned directory using the `cd` command:
   ```bash
   cd llava-1.5-liveness-7b
   ```

### 2. System Requirements
Make sure your system meets the following requirements before proceeding:
- Python 3.10+ is installed on your computer.
- Pip (Python package installer) is installed.


### 3. Create a Virtual Environment
A virtual environment will allow you to separate this project from the global Python installation. Follow these steps to create a virtual environment:

**On Windows:**
Open Command Prompt and enter the following command:
```bash
python -m venv virtualenv_name
```
Replace `virtualenv_name` with the desired name for your virtual environment.

**On macOS and Linux:**
Open the terminal and enter the following command:
```bash
python3 -m venv virtualenv_name
```
Replace `virtualenv_name` with the desired name for your virtual environment.

### 4. Activate the Virtual Environment
After creating the virtual environment, you need to activate it before installing the requirements. Use the following steps:

**On Windows:**
In Command Prompt, enter the following command:
```bash
virtualenv_name\Scripts\activate
```
Replace `virtualenv_name` with the name you provided in the previous step.

**On macOS and Linux:**
In the terminal, enter the following command:
```bash
source virtualenv_name/bin/activate.bat
```
Replace `virtualenv_name` with the name you provided in the previous step.

### 5. Install Requirements
Once the virtual environment is activated, you can install the project requirements from the `requirements.app.txt` file. Follow these steps:

**On Windows, macOS, and Linux:**
In the activated virtual environment, navigate to the directory where the `requirements.app.txt` file is located. Then, enter the following command:
```bash
pip install -r requirements.app.txt
```
This command will install all the required packages specified in the `requirements.app.txt` file 

### 6. Run Gradio

How to run Web App:

``python frontend/app.py``

## Reference

- [Blog Tutorial Finetuning Vision Language Model](https://huggingface.co/blog/vlms) 
- [Base Vision Language Model](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- [Paper Vision Language Model](https://arxiv.org/abs/2304.08485)

