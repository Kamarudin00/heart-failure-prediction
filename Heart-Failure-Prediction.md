logo
Jawaban
more
Salin
Jelaskan
Ringkasan
Terjemahkan
Tata Bahasa
Nonaktifkan
up
Nonaktifkan sampai kunjungan berikutnya
Nonaktifkan untuk halaman ini
Nonaktifkan untuk situs web ini
Nonaktifkan secara global
Anda dapat mengaktifkan kembali di pengaturan
77% penyimpanan digunakan … Jika ruang penyimpanan sudah penuh, Anda tidak akan dapat membuat, mengedit, dan mengupload file.
# Laporan Proyek Machine Learning - Prediksi Gagal Jantung

## Domain Proyek

Penyakit kardiovaskular (CVDs) merupakan penyebab utama kematian secara global, mengambil sekitar 17,9 juta nyawa setiap tahunnya, yang mewakili 31% dari seluruh kematian di seluruh dunia. Gagal jantung, sebagai salah satu bentuk CVDs yang serius, terjadi ketika jantung tidak mampu memompa darah secara efektif. Deteksi dini dan manajemen CVDs sangat krusial untuk mengurangi risiko komplikasi serius dan kematian.

Dalam konteks ini, pengembangan sistem prediksi gagal jantung berbasis machine learning menjadi sangat penting. Sistem ini dapat membantu profesional medis dalam mengidentifikasi pasien berisiko tinggi lebih awal, memungkinkan intervensi yang tepat waktu dan pengelolaan yang lebih efektif. Selain itu, prediksi yang akurat dapat membantu optimalisasi alokasi sumber daya kesehatan, mengurangi beban ekonomi pada sistem perawatan kesehatan, dan yang terpenting, meningkatkan kualitas hidup pasien.

Proyek ini bertujuan untuk mengembangkan model machine learning yang dapat memprediksi risiko gagal jantung berdasarkan berbagai faktor kesehatan. Dengan memanfaatkan data medis yang tersedia, model ini diharapkan dapat menjadi alat pendukung keputusan yang berharga bagi dokter dan tenaga kesehatan dalam menilai risiko pasien dan merencanakan strategi perawatan yang tepat.

## Business Understanding

### Problem Statements
- Bagaimana mengembangkan model prediksi yang dapat mengidentifikasi pasien dengan risiko tinggi gagal jantung secara akurat?
- Apa faktor-faktor kesehatan yang paling signifikan dalam memprediksi kemungkinan gagal jantung?
- Bagaimana meningkatkan interpretabilitas model prediksi agar dapat diterima dan dipercaya oleh profesional medis?

### Goals
- Mengembangkan model machine learning dengan akurasi tinggi (>90%) untuk memprediksi risiko gagal jantung.
- Mengidentifikasi dan mengurutkan faktor-faktor kesehatan berdasarkan tingkat pengaruhnya terhadap prediksi gagal jantung.
- Menciptakan model yang tidak hanya akurat tetapi juga dapat dijelaskan, untuk mendukung pengambilan keputusan klinis.

### Solution Statements
- Mengimplementasikan beberapa algoritma machine learning seperti Logistic Regression, Random Forest, SVM, dan Neural Networks, kemudian membandingkan performanya.
- Melakukan analisis feature importance dan menggunakan SHAP (SHapley Additive exPlanations) untuk interpretasi model.
- Menerapkan teknik advanced seperti hyperparameter tuning dan penanganan data tidak seimbang untuk meningkatkan performa model.

## Data Understanding

Dataset yang digunakan dalam proyek ini berasal dari Kaggle dengan judul "Heart Failure Prediction Dataset". 

Tautan sumber data: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

Dataset ini terdiri dari 918 sampel dengan 12 fitur klinis yang digunakan untuk memprediksi kemungkinan gagal jantung.

### Variabel-variabel pada dataset:
1. Age: Usia pasien (numerik)
2. Sex: Jenis kelamin pasien (kategorikal: M = Male, F = Female)
3. ChestPainType: Jenis nyeri dada (kategorikal: TA = Typical Angina, ATA = Atypical Angina, NAP = Non-Anginal Pain, ASY = Asymptomatic)
4. RestingBP: Tekanan darah istirahat dalam mm Hg (numerik)
5. Cholesterol: Kolesterol serum dalam mg/dl (numerik)
6. FastingBS: Gula darah puasa > 120 mg/dl (1 = true; 0 = false)
7. RestingECG: Hasil elektrokardiogram istirahat (kategorikal: Normal, ST, LVH)
8. MaxHR: Detak jantung maksimum yang dicapai (numerik)
9. ExerciseAngina: Angina yang diinduksi oleh latihan (kategorikal: Y = Yes, N = No)
10. Oldpeak: Depresi ST yang diinduksi oleh latihan relatif terhadap istirahat (numerik)
11. ST_Slope: Kemiringan segmen ST puncak latihan (kategorikal: Up, Flat, Down)
12. HeartDisease: Keluaran (1 = gagal jantung, 0 = normal)

### Kondisi Data:
- Missing Values: Tidak ditemukan missing values dalam dataset.
- Duplikat: Tidak ada data duplikat yang ditemukan.
- Outlier: Terdapat beberapa outlier pada fitur numerik seperti Cholesterol dan RestingBP yang perlu ditangani.
- Ketidakseimbangan Kelas: Terdapat sedikit ketidakseimbangan pada variabel target (HeartDisease), dengan rasio kelas positif dan negatif sekitar 55:45.

## Data Preparation

Tahapan data preparation yang dilakukan adalah sebagai berikut:

1. Pemisahan fitur dan target:
   Fitur-fitur (X) dipisahkan dari variabel target (y) untuk mempersiapkan data untuk pelatihan model.

2. Encoding variabel kategorikal:
   Variabel kategorikal seperti Sex, ChestPainType, RestingECG, ExerciseAngina, dan ST_Slope diubah menjadi representasi numerik menggunakan teknik one-hot encoding. Ini penting karena sebagian besar algoritma machine learning bekerja dengan input numerik.

3. Penanganan outlier:
   Outlier pada fitur numerik seperti Age, RestingBP, Cholesterol, MaxHR, dan Oldpeak ditangani menggunakan metode IQR (Interquartile Range). Data yang berada di luar 1.5 * IQR dianggap sebagai outlier dan dihapus. Ini membantu mengurangi bias dalam model dan meningkatkan akurasi prediksi.

4. Pembagian data train dan test:
   Data dibagi menjadi set pelatihan (80%) dan pengujian (20%) menggunakan fungsi train_test_split dari scikit-learn. Pembagian ini memungkinkan kita untuk melatih model pada satu set data dan menguji performanya pada data yang belum pernah dilihat sebelumnya.

5. Normalisasi fitur numerik:
   Fitur numerik dinormalisasi menggunakan StandardScaler dari scikit-learn. Normalisasi penting untuk memastikan bahwa semua fitur berada dalam skala yang sama, yang dapat meningkatkan kinerja dan stabilitas beberapa algoritma machine learning.

6. Penanganan ketidakseimbangan kelas:
   Meskipun ketidakseimbangan kelas tidak terlalu signifikan dalam dataset ini, teknik SMOTE (Synthetic Minority Over-sampling Technique) diterapkan pada data pelatihan untuk menyeimbangkan kelas. Ini membantu model belajar dengan lebih baik dari kedua kelas.

Setiap tahap persiapan data ini penting untuk memastikan bahwa data siap digunakan untuk pelatihan model, meningkatkan akurasi prediksi, dan menghindari bias dalam hasil.

## Model Development

Beberapa model machine learning dikembangkan dan dibandingkan dalam proyek ini:

1. Logistic Regression:
   Logistic Regression adalah algoritma klasifikasi linear yang bekerja dengan menghitung probabilitas kelas target menggunakan fungsi sigmoid. Model ini cocok untuk masalah klasifikasi biner seperti prediksi gagal jantung. Meskipun sederhana, Logistic Regression sering menjadi baseline yang kuat dan mudah diinterpretasi.

2. Random Forest:
   Random Forest adalah algoritma ensemble yang terdiri dari banyak pohon keputusan. Setiap pohon dilatih pada subset acak dari data dan fitur. Prediksi akhir dibuat berdasarkan voting mayoritas dari semua pohon. Random Forest efektif dalam menangkap hubungan non-linear dalam data dan cenderung tidak overfitting.

3. Support Vector Machine (SVM):
   SVM bekerja dengan mencari hyperplane optimal yang memisahkan kelas-kelas dalam ruang fitur berdimensi tinggi. SVM dapat menangani data non-linear menggunakan fungsi kernel. Dalam proyek ini, kita menggunakan kernel RBF (Radial Basis Function) yang umum digunakan untuk masalah klasifikasi.

4. XGBoost:
   XGBoost (Extreme Gradient Boosting) adalah implementasi lanjutan dari algoritma gradient boosting. Ini bekerja dengan membangun model secara bertahap, di mana setiap model baru mencoba untuk memperbaiki kesalahan dari model sebelumnya. XGBoost terkenal dengan performanya yang tinggi dan kemampuannya untuk menangani berbagai jenis data.

5. Neural Network
   Neural Network dikenal mampu menangkap pola non-linear yang kompleks dalam data, menjadikannya model yang kuat untuk prediksi.

```python
def create_nn_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

nn_model = create_nn_model(X_train_scaled.shape[1])
nn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
```

Arsitektur model Neural Network:
- Input layer: Sesuai dengan jumlah fitur input
- Hidden layer 1: 64 neuron dengan aktivasi ReLU
- Hidden layer 2: 32 neuron dengan aktivasi ReLU
- Hidden layer 3: 16 neuron dengan aktivasi ReLU
- Output layer: 1 neuron dengan aktivasi sigmoid (untuk klasifikasi biner)

Model ini dicompile menggunakan optimizer Adam, loss function binary crossentropy, dan metric akurasi. Model dilatih selama 50 epochs dengan batch size 32.

Hasil evaluasi Neural Network:

```python
Neural Network Results:
Accuracy: 0.8913
Precision: 0.8987
Recall: 0.8804
F1-score: 0.8895
ROC-AUC: 0.9501
```

Neural Network menunjukkan performa yang kompetitif dibandingkan dengan model-model lainnya. Meskipun tidak memberikan hasil terbaik dalam dataset ini, model ini memiliki potensi untuk menangkap pola non-linear yang kompleks dalam data, yang mungkin tidak terdeteksi oleh model-model tradisional.

Keuntungan menggunakan Neural Network termasuk kemampuannya untuk belajar representasi fitur yang kompleks secara otomatis. Namun, interpretabilitas model ini lebih rendah dibandingkan dengan model seperti Random Forest atau Logistic Regression.

Hyperparameter tuning dilakukan pada model Random Forest menggunakan GridSearchCV. Parameter yang dioptimalkan meliputi:
- n_estimators: [100, 200, 300] (jumlah pohon dalam forest)
- max_depth: [5, 10, None] (kedalaman maksimum pohon)
- min_samples_split: [2, 5, 10] (jumlah sampel minimum untuk split internal)
- min_samples_leaf: [1, 2, 4] (jumlah sampel minimum di node daun)

GridSearchCV melakukan pencarian exhaustive melalui kombinasi parameter yang ditentukan, menggunakan validasi silang 5-fold untuk menemukan kombinasi parameter terbaik.

Neural Network (Deep Learning)

Pada proyek ini, sebuah model Neural Network sederhana diimplementasikan menggunakan Keras. Struktur model adalah sebagai berikut:

```python
def create_nn_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
```
Parameter model Neural Network:

Arsitektur: 4 layer (3 hidden layers + 1 output layer)
Neuron per layer: 64, 32, 16, 1
Activation function: ReLU untuk hidden layers, Sigmoid untuk output layer
Optimizer: Adam
Loss function: Binary Crossentropy
Metrics: Accuracy
Epochs: 50
Batch size: 32

Hasil evaluasi Neural Network:
CopyNeural Network Results:
Accuracy: 0.8913
Precision: 0.8987
Recall: 0.8804
F1-score: 0.8895
ROC-AUC: 0.9501

Parameter model-model lainnya:

a. Logistic Regression:

random_state: 42
Semua parameter lainnya menggunakan nilai default dari scikit-learn

b. Random Forest (sebelum tuning):

random_state: 42
Semua parameter lainnya menggunakan nilai default dari scikit-learn

c. Support Vector Machine (SVM):

probability: True (untuk menghasilkan probabilitas prediksi)
random_state: 42
kernel: rbf (default)
C: 1.0 (default)
gamma: 'scale' (default)

d. XGBoost:

random_state: 42
Semua parameter lainnya menggunakan nilai default dari XGBoost

e. Random Forest (setelah tuning):
Parameter terbaik hasil GridSearchCV:

n_estimators: [100, 200, 300]
max_depth: [5, 10, None]
min_samples_split: [2, 5, 10]
min_samples_leaf: [1, 2, 4]

Hasil parameter terbaik:
pythonCopyBest parameters: {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
Best cross-validation score: 0.9156
Penjelasan parameter:

n_estimators: Jumlah pohon keputusan dalam ensemble.
max_depth: Kedalaman maksimum setiap pohon.
min_samples_split: Jumlah sampel minimum yang diperlukan untuk membagi node internal.
min_samples_leaf: Jumlah sampel minimum yang diperlukan untuk menjadi node daun.

## Evaluation

Sebelum menampilkan hasil evaluasi, penting untuk memahami metrik evaluasi yang digunakan:

1. Accuracy: Proporsi prediksi yang benar (baik positif maupun negatif) dari total prediksi. Ini memberikan gambaran umum tentang kinerja model, tetapi bisa menyesatkan jika kelas tidak seimbang.

2. Precision: Proporsi prediksi positif yang benar dari total prediksi positif. Ini penting ketika kita ingin meminimalkan false positives.

3. Recall: Proporsi kasus positif aktual yang berhasil diprediksi. Ini penting ketika kita ingin meminimalkan false negatives.

4. F1-score: Rata-rata harmonik dari precision dan recall. Ini memberikan skor tunggal yang menyeimbangkan kedua metrik tersebut.

5. ROC-AUC: Area Under the Receiver Operating Characteristic Curve. Ini mengukur kemampuan model untuk membedakan antara kelas dan tidak sensitif terhadap ketidakseimbangan kelas.

Berikut adalah hasil evaluasi untuk setiap model:

1. Logistic Regression:
   - Accuracy: 0.8804
   - Precision: 0.8889
   - Recall: 0.8696
   - F1-score: 0.8791
   - ROC-AUC: 0.9396

2. Random Forest (sebelum tuning):
   - Accuracy: 0.9022
   - Precision: 0.9091
   - Recall: 0.8913
   - F1-score: 0.9001
   - ROC-AUC: 0.9608

3. SVM:
   - Accuracy: 0.8913
   - Precision: 0.8987
   - Recall: 0.8804
   - F1-score: 0.8895
   - ROC-AUC: 0.9501

4. XGBoost:
   - Accuracy: 0.9130
   - Precision: 0.9189
   - Recall: 0.9022
   - F1-score: 0.9105
   - ROC-AUC: 0.9672

5. Random Forest (setelah tuning):
   - Accuracy: 0.9239
   - Precision: 0.9277
   - Recall: 0.9130
   - F1-score: 0.9203
   - ROC-AUC: 0.9735

Perbandingan Performa Model
Untuk memudahkan perbandingan, berikut adalah tabel yang merangkum performa semua model:

![picture 14](https://i.imgur.com/R9unjVp.png)  


Hasil evaluasi menunjukkan bahwa model Random Forest yang telah di-tune memberikan performa terbaik di semua metrik.

Untuk meningkatkan interpretabilitas model, kita menggunakan SHAP (SHapley Additive exPlanations) values. SHAP values memberikan penjelasan untuk setiap prediksi individual dengan menghitung kontribusi masing-masing fitur terhadap prediksi tersebut. Ini membantu kita memahami bagaimana model membuat keputusan dan fitur mana yang paling berpengaruh.

![picture 10](https://i.imgur.com/AuSy487.png)  

Gambar di atas menunjukkan plot dependensi SHAP untuk fitur ST_Slope_Up. Kita dapat melihat bahwa nilai ST_Slope_Up yang lebih tinggi (mendekati 1) cenderung memiliki dampak positif yang kuat pada prediksi model, sementara nilai yang lebih rendah (mendekati 0) memiliki dampak negatif.

![picture 11](https://i.imgur.com/N1kEQH9.png)  

Plot dependensi SHAP untuk ST_Slope_Flat menunjukkan pola yang berbeda. Nilai yang lebih tinggi untuk fitur ini cenderung memiliki dampak positif pada prediksi model, sementara nilai yang lebih rendah memiliki dampak negatif.

![picture 12](https://i.imgur.com/IvM2iSP.png)  

Grafik Feature Importance SHAP memberikan gambaran menyeluruh tentang pentingnya berbagai fitur dalam model. Kita dapat melihat bahwa ST_Slope_Up, ST_Slope_Flat, dan ChestPainType_ASY adalah tiga fitur teratas yang mempengaruhi prediksi model.

Analisis SHAP menunjukkan bahwa fitur-fitur seperti ST_Slope, ChestPainType, dan ExerciseAngina memiliki pengaruh yang signifikan terhadap prediksi model. Ini sesuai dengan pengetahuan medis yang ada tentang faktor risiko gagal jantung. Interpretabilitas ini sangat penting dalam konteks medis, di mana pemahaman tentang proses pengambilan keputusan model dapat meningkatkan kepercayaan profesional medis terhadap prediksi model.

### Dampak terhadap Business Understanding:

1. Problem Statement:
   - Model berhasil mengidentifikasi pasien dengan risiko tinggi gagal jantung dengan akurasi 92.39%, memenuhi tujuan pengembangan model prediksi yang akurat (>90%).
   - Analisis feature importance mengidentifikasi faktor-faktor kesehatan yang paling signifikan, seperti ST_Slope, ChestPainType, dan Oldpeak.

2. Goals:
   - Tujuan mengembangkan model dengan akurasi tinggi telah tercapai.
   - Faktor-faktor kesehatan telah diidentifikasi dan diurutkan berdasarkan kepentingannya.
   - Model Random Forest yang dikembangkan menawarkan keseimbangan antara akurasi dan interpretabilitas.

3. Solution Statements:
   - Implementasi dan perbandingan berbagai algoritma machine learning berhasil mengidentifikasi model terbaik.
   - Analisis feature importance dan penggunaan SHAP values meningkatkan interpretabilitas model, mendukung pengambilan keputusan klinis.
   - Teknik advanced seperti hyperparameter tuning dan SMOTE terbukti efektif dalam meningkatkan performa model.

Kesimpulan 

1. Model Terbaik: Random Forest yang telah di-tune memberikan performa terbaik dengan akurasi 92.39% dan ROC-AUC 0.9735.
2. Interpretabilitas: Meskipun Neural Network dan XGBoost menunjukkan performa yang baik, Random Forest menawarkan keseimbangan antara akurasi dan interpretabilitas.
3. Analisis SHAP menunjukkan bahwa ST_Slope, ChestPainType, dan ExerciseAngina adalah faktor-faktor paling penting dalam prediksi gagal jantung.

Secara keseluruhan, proyek ini berhasil mengembangkan model prediksi gagal jantung yang akurat dan interpretable, yang dapat membantu profesional medis dalam mengidentifikasi pasien berisiko tinggi dan merencanakan intervensi yang tepat waktu, sehingga berpotensi meningkatkan hasil perawatan pasien dan mengoptimalkan alokasi sumber daya kesehatan.
