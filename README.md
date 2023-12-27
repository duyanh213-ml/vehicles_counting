<center>
    <h1>
        VEHICLES COUTING SYSTEM
    </h1>
</center>

### Objective

In this repository, we've developed a system for detecting vehicles in highway camera videos. 

### Repo structure
```
Vehicles_counting_system
├── data_preparation  
├── dataset
├── for_readme
├── draft
├── runs
├── train
├── vehicles_couting_system
├── video
└── LICENSE
```

After cloning this repository, you will notice the absence of two folders: `dataset` and `video`. To rectify this, create a folder named `\video` and obtain both the videos and the dataset folder's zip file from <a href="https://drive.google.com/drive/folders/111ssjhY7VyAcEarYErcIaeyf-C25FgK8?usp=sharing">here</a>

- #### Data preparation
    We create images for the Yolo model using three 'yolo-videos' located in the `\video` directory (`for_yolo_part1`, `for_yolo_part2`, `for_yolo_part3`). Next, we divide them into training and validation sets. To prepare for our training, we utilize <a href="https://www.cvat.ai/">CVAT</a> to annotate all the images.

    You have the option to run the file `\data_preparation\data_prep.py` to generate a new image dataset and label them yourself. Alternatively, you can use the <a href="https://drive.google.com/drive/folders/111ssjhY7VyAcEarYErcIaeyf-C25FgK8?usp=sharing">dataset</a> that I've already created.

- #### Training
    The training process is constructed according to the Ultralytics documentation for <a href="https://docs.ultralytics.com/modes/train/">YOLOv8</a>.

### System architecture

This system comprises two main components. Initially, we trained a pre-trained YOLOv8 model using our custom dataset, employing it as a detector to identify vehicles. The second component involves a tracking algorithm (KCF-Tracking) utilized to track the detected objects. Subsequently, the detector re-initiates detection after approximately every 10 frames (modifiable based on preference). Finally, we employ a laser line to count the number of vehicles passing through it.

![Alt text](/for_readme/vehicles_counting_system.png)

To run the system you just need to run the `\vehicles_counting_system\main.py`

```
python \vehicles_counting_system\main.py
```

### Python version

```bash
Python -V 3.10.12
```

### Note for requirements
If you encounter errors while installing the packages listed in `requirements.txt`, simply remove the problematic package from `requirements.txt` and then rerun `pip install -r requirements.txt.` Specifically for PyTorch, consider installing it directly by running this command:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
