# Models Directory

This directory contains pre-trained models required for the Face Shape Detection project.

---

## 📁 Required Files

### 1. shape_predictor_68_face_landmarks.dat

**Purpose:** Detects 68 facial landmark points for face shape analysis

**Download:**
1. Visit: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
2. Download the `.bz2` file
3. Extract using:
   ```bash
   # Windows (using 7-Zip or WinRAR)
   # Or online: https://extract.me/

   # macOS/Linux:
   bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
   ```
4. Place the extracted `.dat` file in this `models/` directory

**File Size:** ~99 MB (compressed), ~99 MB (extracted)

**License:** Free for non-commercial use

---

### 2. KNN_Model.sav

**Purpose:** Pre-trained KNN classifier for face shape classification

**This file is NOT included in the repository** because it needs to be trained on your own dataset or obtained separately.

#### Option A: Train Your Own Model

Create a training script:

```python
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Your training data
# X = face features (distance ratios)
# y = face shape labels (0-5)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# Save model
joblib.dump(knn, 'models/KNN_Model.sav')
```

#### Option B: Use a Pre-trained Model

If you have access to a pre-trained model:
1. Obtain the `KNN_Model.sav` file
2. Place it in this `models/` directory

**Expected Features:**
- Input: 5 normalized distance ratios (d1, d2, d3, d4, d5)
- Output: Face shape label (0-5)
  - 0: Diamond
  - 1: Oblong
  - 2: Oval
  - 3: Round
  - 4: Square
  - 5: Triangle

---

## 🔒 .gitignore

Model files are **excluded from Git** because:
- Large file sizes
- License restrictions
- Personal training data

The `.gitignore` includes:
```
*.dat
*.sav
*.h5
*.pkl
models/*.dat
models/*.sav
```

---

## ✅ Verification

After downloading/creating model files, your `models/` directory should look like:

```
models/
├── README.md  (this file)
├── shape_predictor_68_face_landmarks.dat  (~99 MB)
└── KNN_Model.sav  (~1-5 KB)
```

Verify files exist:
```bash
ls -lh models/
```

---

## 🚨 Troubleshooting

### "FileNotFoundError: shape_predictor_68_face_landmarks.dat"

**Solution:**
1. Verify file is in `models/` directory
2. Check filename (no extra extensions like `.bz2`)
3. Ensure file is ~99 MB (fully extracted)

### "FileNotFoundError: KNN_Model.sav"

**Solution:**
1. Train your own model (see Option A above)
2. Or obtain a pre-trained model
3. Place in `models/` directory

### "dlib.error: Unable to open file"

**Solution:**
- File might be corrupted during download
- Re-download and extract again
- Ensure sufficient disk space

---

## 📚 Alternative Sources

### Shape Predictor Model

Official dlib models:
- http://dlib.net/files/

Alternative hosts:
- GitHub releases of dlib projects
- Academic repositories

### Training Your Own KNN Model

Required dataset structure:
```csv
d1,d2,d3,d4,d5,face_shape
85.2,92.1,98.3,105.7,112.4,0
78.9,88.2,95.6,103.8,110.2,1
...
```

Minimum recommended samples:
- At least 50 samples per face shape class
- Total: 300+ samples for 6 classes

---

## 🔗 Resources

- **dlib Documentation:** http://dlib.net/
- **Face Shape Datasets:** Search on Kaggle or academic datasets
- **Model Training Tutorial:** See `docs/TRAIN_FACE_MODEL.md` (if available)

---

## 📝 Notes

- Model files are for **non-commercial, educational use only**
- Check license requirements for production use
- Always credit original authors
- Consider privacy when using face data

---

## ⚠️ Important

**Never commit model files to Git:**
- They're large and slow down repository
- May violate license terms
- Keep them local or use Git LFS for large files

---

*Last Updated: November 2024*
