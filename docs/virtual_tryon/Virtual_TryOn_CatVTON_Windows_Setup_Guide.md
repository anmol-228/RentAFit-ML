# RentAFit Virtual Try-On Setup Guide

## CatVTON on a Blank Windows NVIDIA PC

Prepared for: RentAFit project  
Checked against official sources on: March 20, 2026

---

## 1. Purpose of This Guide

This guide explains, step by step, how to run **CatVTON** on a Windows PC that currently has **nothing installed for coding**.

This is written for a beginner. It assumes:
- the PC is running Windows 10 or Windows 11
- the PC has an NVIDIA GPU
- the user is not comfortable with Windows terminal setup yet
- the goal is to **run virtual try-on locally**, not train a new model

This guide focuses on **inference only**.

---

## 2. What CatVTON Will Do for the Project

CatVTON is the virtual try-on model chosen for the RentAFit project.

It takes:
- a **person image**
- a **garment image**

and generates:
- an output image showing the person wearing the garment

For the current RentAFit project scope, the best starting categories are:
- Shirt
- Top
- Jacket

These categories are simpler and more reliable than saree, lehenga, or more heavily draped clothing.

---

## 3. Why This Guide Uses the Official CatVTON App Route

This guide uses the **official CatVTON repository** and its **local Gradio app**.

This is the best first attempt because:
- the official repository documents this workflow directly
- the official app can auto-download checkpoints on first run
- the project notes that DensePose and SCHP were localized to reduce environment issues
- this is cleaner for a beginner than starting with custom ComfyUI node debugging

This is an informed recommendation based on the official project documentation.

---

## 4. Very Important Warning Before You Start

If the Windows PC has one of these GPUs:
- RTX 5080
- RTX 5090

stop before installing and re-check the environment setup, because very new GPUs sometimes need slightly different PyTorch or CUDA handling.

If the GPU is one of these, this guide is usually appropriate:
- RTX 3060 / 3070 / 3080 / 3090
- RTX 4060 / 4070 / 4080 / 4090

---

## 5. What You Need Before Installation

Before you start, make sure you have:
- a stable internet connection
- at least 30 GB of free disk space
- an NVIDIA GPU with enough VRAM
- clean test images ready
- enough time for downloads, because the first run can take quite a while

The official CatVTON project notes that `1024 x 768` inference with `bf16` can run in under `8 GB` VRAM.

In practice, more VRAM is still better.

---

## 6. Prepare the Test Images First

Before installing anything, prepare a small clean test set.

### 6.1 Person images

Prepare at least 5 person photos.

Each should be:
- front-facing
- upper body clearly visible
- well lit
- not blurry
- with a simple or clean background
- with the torso not blocked heavily by arms, bags, or scarves

### 6.2 Garment images

Prepare at least 5 garment photos.

Each should be:
- a single garment only
- front-facing if possible
- clearly visible
- not folded badly
- well lit
- not a collage
- not a screenshot with text over it

### 6.3 Categories to start with

Use only:
- Shirt
- Top
- Jacket

Do not start with:
- Saree
- Lehenga
- heavily layered ethnic sets

---

## 7. Create Folders on the Windows PC

Create a clean folder structure first.

Example root folder:

```text
D:\RentAFit_TryOn
```

Inside that folder, create:

```text
D:\RentAFit_TryOn\inputs\person
D:\RentAFit_TryOn\inputs\garment
D:\RentAFit_TryOn\outputs
D:\RentAFit_TryOn\notes
D:\RentAFit_TryOn\software
```

Put your test images here:

```text
D:\RentAFit_TryOn\inputs\person\person_01.jpg
D:\RentAFit_TryOn\inputs\person\person_02.jpg
D:\RentAFit_TryOn\inputs\garment\shirt_01.jpg
D:\RentAFit_TryOn\inputs\garment\top_01.jpg
D:\RentAFit_TryOn\inputs\garment\jacket_01.jpg
```

Use simple names. Avoid spaces and strange symbols.

---

## 8. Check the GPU Model on Windows

Do this before anything else.

### Method

1. Press `Ctrl + Shift + Esc`
2. Open **Task Manager**
3. Click **Performance**
4. Click **GPU**
5. Note the exact GPU model name

Write this down in your notes.

Example:
- NVIDIA GeForce RTX 4070

If the GPU is 30-series or 40-series, this guide is usually fine.

---

## 9. Update the NVIDIA Driver

This step is very important.

### What to do

1. Open the official NVIDIA driver page:
   [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
2. Select the exact GPU model
3. Download the latest driver
4. Install it
5. Restart the computer

Do not skip the restart.

---

## 10. Install Miniconda

The official CatVTON setup uses a conda environment, so this guide uses **Miniconda**.

### Official links

- [Miniconda install guide](https://www.anaconda.com/docs/getting-started/miniconda/install)
- [Miniconda download page](https://www.anaconda.com/download)

### What to do

1. Open the Miniconda download page
2. Download the **Windows 64-bit installer**
3. Run the installer
4. Choose **Just Me** unless you specifically need all users
5. Finish installation
6. Open **Anaconda Prompt** from the Start menu

Use **Anaconda Prompt** for the rest of the setup.

Do not worry about regular Command Prompt for now.

---

## 11. Download the Official CatVTON Repository

### Official source

- [CatVTON official GitHub repository](https://github.com/Zheng-Chong/CatVTON)

### What to do

1. Open the GitHub page
2. Click the green **Code** button
3. Click **Download ZIP**
4. Save the ZIP file into:

```text
D:\RentAFit_TryOn\software
```

5. Extract it
6. Rename the extracted folder to:

```text
D:\RentAFit_TryOn\software\CatVTON
```

Avoid spaces in the folder path.

---

## 12. Open Anaconda Prompt and Create the Environment

Open **Anaconda Prompt**.

Then run this command:

```bash
conda create -n catvton python=3.9.0 -y
```

Wait for it to finish.

Then activate the environment:

```bash
conda activate catvton
```

You should now see `(catvton)` at the beginning of the command line.

---

## 13. Move Into the CatVTON Folder

Still inside **Anaconda Prompt**, go to the extracted CatVTON folder:

```bash
cd /d D:\RentAFit_TryOn\software\CatVTON
```

The `/d` is important on Windows because it changes the drive as well.

---

## 14. Install GPU PyTorch First

The official CatVTON requirements pin these versions:
- `torch==2.1.2`
- `torchvision==0.16.2`

To make sure the install uses the NVIDIA GPU properly, install PyTorch first using the official PyTorch command.

### Official source

- [PyTorch previous versions page](https://pytorch.org/get-started/previous-versions)

### Command

```bash
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

Wait for this to finish fully.

---

## 15. Install the Remaining CatVTON Requirements

Now install the rest of the project dependencies.

Inside the same Anaconda Prompt window, still in the CatVTON folder, run:

```bash
pip install -r requirements.txt
```

### Official requirements reference

- [CatVTON requirements.txt](https://raw.githubusercontent.com/Zheng-Chong/CatVTON/main/requirements.txt)

This step can also take some time.

---

## 16. Verify That CUDA Is Available

Before running the app, check whether Python can actually see the GPU.

Run this command:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### Expected result

You want to see something like:
- `CUDA available: True`
- `GPU count: 1`
- your GPU name printed correctly

If you get `False`, do not continue yet.

---

## 17. Start the Official CatVTON App

The official CatVTON README provides this command for the local app:

```bash
python app.py --output_dir="resource/demo/output" --mixed_precision="bf16" --allow_tf32
```

### Important first-run note

The official repository says the app can auto-download the checkpoints the first time it runs.

That means:
- first startup can be slow
- downloads can be large
- it can take many minutes

Do not assume it is broken too early.

---

## 18. If `bf16` Fails, Use `fp16`

If the app gives a precision-related error, close it and run this instead:

```bash
python app.py --output_dir="resource/demo/output" --mixed_precision="fp16" --allow_tf32
```

This is the safer fallback.

---

## 19. Open the Local Web App

Once the app starts successfully, the terminal should print a local web address.

Usually it will look something like:
- `http://127.0.0.1:7860`

Open the exact address printed in the terminal in the browser on that same PC.

---

## 20. Run Your First Test Only

Do not try many images first.

Use only:
- 1 person image
- 1 garment image

Example first test:
- `person_01.jpg`
- `shirt_01.jpg`

### Inside the app

1. upload the person image
2. upload the garment image
3. leave most settings at default for the first run
4. generate the try-on result

The goal of this first test is simple:
- prove the pipeline runs successfully

---

## 21. Save the Output Properly

When the first run completes:

1. save the generated output image
2. copy or move it into:

```text
D:\RentAFit_TryOn\outputs
```

Use a clean filename such as:

```text
test_01_person_01_shirt_01.png
```

Also create a text or spreadsheet note in:

```text
D:\RentAFit_TryOn\notes
```

Suggested columns:
- `test_id`
- `person_image`
- `garment_image`
- `category`
- `status`
- `quality_note`

Example:

```text
T001, person_01.jpg, shirt_01.jpg, Shirt, success, Good overall fit, sleeve edges slightly soft
```

---

## 22. Run 5 to 10 Small Tests After the First Success

Once the first run works, test a small batch.

Recommended order:
- 2 Shirt cases
- 2 Top cases
- 1 Jacket case

Then expand if results look good.

This is enough for:
- project documentation
- demonstration
- comparison of good and bad results

---

## 23. Troubleshooting: CUDA Is Not Available

If the verification command says:
- `CUDA available: False`

then check these in order:

1. Did you update the NVIDIA driver?
2. Did you restart the PC?
3. Did you open **Anaconda Prompt**, not normal terminal?
4. Does the prompt show `(catvton)`?
5. Did the PyTorch install command complete successfully?

If this still fails, note the exact error and stop there.

---

## 24. Troubleshooting: Out of Memory

If you get an out-of-memory error:

1. close Chrome tabs and other heavy apps
2. rerun using `fp16` instead of `bf16`
3. if the app offers output size settings, reduce the size for testing
4. do not run multiple GPU-heavy tools at the same time

---

## 25. Troubleshooting: First Run Takes Too Long

This is often normal.

Possible reasons:
- checkpoint download
- slow internet
- initial model loading
- large dependency initialization

If the terminal is still active and not crashing, wait longer before interrupting it.

---

## 26. Troubleshooting: Output Quality Is Poor

Poor output quality is often caused by poor inputs, not broken installation.

Check whether:
- the person image is too dark
- the garment image is cluttered
- the garment is not clearly visible
- the pose is awkward
- the torso is partly blocked
- the photo is blurry

Before changing code, improve the images.

---

## 27. What You Should Not Do Right Now

Do not:
- train CatVTON from scratch
- start with saree or lehenga
- jump into website integration before the local demo works
- test too many categories at once
- use random bad-quality images first

Keep the first milestone small and controlled.

---

## 28. Recommended Documentation You Should Keep

For your college project, save these:
- GPU model used
- RAM size if known
- number of test cases
- successful outputs
- failed outputs
- observed strengths
- observed failure cases

This will help later in the final report and viva.

---

## 29. Best Workflow Summary

Use this exact order:

1. check GPU model
2. update NVIDIA driver
3. create folders
4. prepare 5 person images and 5 garment images
5. install Miniconda
6. download CatVTON ZIP
7. create the `catvton` environment
8. install PyTorch with CUDA support
9. install CatVTON requirements
10. verify CUDA is available
11. run the app
12. do 1 test
13. save the result
14. do 5 to 10 tests
15. document everything

---

## 30. Fallback Option If This Fails on Windows

If the official `app.py` route fails badly on Windows, the fallback is:
- ComfyUI + CatVTON

Official references for that route:
- [ComfyUI official docs](https://docs.comfy.org/)
- [Official CatVTON Windows note issue #8](https://github.com/Zheng-Chong/CatVTON/issues/8)

Use this fallback only if the direct app route becomes unmanageable.

---

## 31. License Note

The CatVTON model page lists the materials under:
- `CC BY-NC-SA 4.0`

That means it is suitable for a non-commercial academic project, but not something to assume is unrestricted for commercial use.

Official source:
- [CatVTON Hugging Face page](https://huggingface.co/zhengchong/CatVTON)

---

## 32. Official Sources Used

- [CatVTON official GitHub](https://github.com/Zheng-Chong/CatVTON)
- [CatVTON official Hugging Face page](https://huggingface.co/zhengchong/CatVTON)
- [Miniconda official install guide](https://www.anaconda.com/docs/getting-started/miniconda/install)
- [PyTorch previous versions page](https://pytorch.org/get-started/previous-versions)
- [CatVTON Windows note issue #8](https://github.com/Zheng-Chong/CatVTON/issues/8)
- [ComfyUI official docs](https://docs.comfy.org/)

---

## 33. Final Note

The correct goal for the first session on your friend’s PC is not perfection.

The correct goal is:
- get one clean successful try-on run
- save it properly
- then expand carefully

Once that first success is done, the rest becomes much easier.
