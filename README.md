License and github Open source are referenced to https://github.com/yangsenius/TransPose.
Paper: S. Yang, Z. Quan, M. Nie, and W. Yang, “TransPose: Keypoint Localization via Transformer”, Accessed: Aug. 26, 2024. [Online]. Available: https://github.com/yangsenius/TransPose

### 1. Set Up the Environment

#### For the Django Back-end
1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```
   
2. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
     
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
4. **Modify the local directory**:
   - Go to /custorm_scripts/runserver.py.
   - Modify the "root_dir" path corresponding with your local dir.
   - By default, the meta-updated pretrained model is based on TransPose R-A4 model. Hence, we need to modify the config file for TransPose R-A4 by modify   
     /experiments/coco/transpose_r/TP_R_256x192_d256_h1024_enc4_mh8.yaml. Modify DATASET.ROOT : '<your_local_dir>/data/coco/'.

     
5. **Run Server**
   ```bash
   cd /lib
   python ~/custom_scripts/runserver.py
   ```
